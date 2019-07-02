from typing import Any, Dict, List, Union
from overrides import overrides
import logging

import gzip
import json
import re

from allennlp.common.registrable import Registrable


## Some utility functions

def unique(data, key=None):
    key = key or (lambda x: x)
    seen = set()
    res = []
    for d in data:
        dkey = key(d)
        if not dkey in seen:
            seen.add(dkey)
            res.append(d)
    return res


def basic_string_hash(string):
    return "".join(re.findall(r'[a-z]+', string.lower()))


def combine_sentences(hits, num: int = None, max_len: int = None) -> str:
    # Assume already sorted by score
    sentences = [hit["text"] for hit in hits]
    sentences = unique(sentences, basic_string_hash)
    if max_len is not None:
        sentences = list(filter(lambda x: len(x) <= max_len, sentences))
    if num is not None:
        sentences = sentences[:num]
    sentences_processed = []
    for sentence in sentences:
        new = sentence.strip()
        if re.match(r'.*\w$', new):
            new = new + "."
        sentences_processed.append(new)
    output = " ".join(reversed(sentences_processed))
    return output


def save_jsonl_gz(file_name, data):
    output = "\n".join(json.dumps(x) for x in data)
    with gzip.open(file_name, "wb") as file:
        file.write(output.encode("utf-8"))


def load_jsonl_gz(file_name):
    with gzip.open(file_name, "rb") as file:
        data_raw = file.read()
    data = data_raw.decode("utf-8")
    return [json.loads(x) for x in data.split("\n")]

##


class DocumentRetriever(Registrable):

    def __init__(self,
                 cache_file_out: str = None,
                 cache_files: str = None,
                 cache_format: str = "jsonl.gz",
                 cache_save_frequency: int = None,
                 max_cache_size: int = None):

        self._cache_file_out = cache_file_out
        self._cache_files = cache_files
        self._cache_format = cache_format
        self._use_cache = False
        if cache_file_out or cache_files:
            self._use_cache = True
        self._cache = {}
        if cache_files:
            self._load_cache_files(cache_files)
        self._max_cache_size = max_cache_size
        self._cache_save_frequency = cache_save_frequency
        self._cache_fill_counter = 0

    def query(self, query: Dict[str, str]):
        cache_key = self._cache_key(query)
        if cache_key in self._cache:
            raw_res = self._cache[cache_key]
        else:
            raw_res = self._retriever(query)
            if raw_res is not None:
                self._cache_add(cache_key, raw_res)
        res = raw_res
        return res

    def query_simple(self, text: str):
        return self.query(query={"text": text})

    def _retriever(self, query: Dict[str, str]):
        return NotImplementedError

    def _cache_key(self, query: Dict[str, str]):
        res = ""
        for key in sorted(query.keys()):
            res += f"${key}$:{query[key]};"
        return res

    def _cache_add(self, cache_key, result):
        if self._max_cache_size and len(self._cache) >= self._max_cache_size:
            return
        else:
            self._cache[cache_key] = result
            self._cache_fill_counter += 1
            if self._cache_save_frequency is not None \
                    and self._cache_fill_counter % self._cache_save_frequency == 0:
                self.save_cache_file()


    def _load_cache_files(self, cache_files):
        cache_files = cache_files.split(",")
        logging.info(f"Loading document_retriever caches from {cache_files}")
        assert(self._cache_format == 'jsonl.gz')
        for cache_file in cache_files:
            data = load_jsonl_gz(cache_file)
            for item_json in data:
                self._cache[item_json["key"]] = item_json["res"]
        logging.info(f"Size of document_retriever caches is {len(self._cache)}")

    def save_cache_file(self, file_name=None):
        file_name = file_name or self._cache_file_out
        if file_name is not None:
            data = [{"key": key, "res": res} for key, res in self._cache.items()]
            save_jsonl_gz(file_name, data)


@DocumentRetriever.register("elastic_search_qa")
class ElasticSearchQARetriever(DocumentRetriever):
    """
    Use ElasticSearch to query based on question ('q' key) and optional answer ('a').
    """

    def __init__(self,
                 host: str,
                 port: int = 9200,
                 indices: Union[str, List[str]]= None,
                 query_format: str = "aristo-qa",
                 retries: int = 3,
                 timeout: int = 60,
                 max_question_length: int = None,
                 max_document_length: int = None,
                 num_retrievals: int = 100,
                 cache_file_out: str = None,
                 cache_files: str = None,
                 cache_format: str = "jsonl.gz",
                 cache_save_frequency: int = None,
                 max_cache_size: int = None):

        from elasticsearch import Elasticsearch

        super().__init__(cache_file_out=cache_file_out,
                         cache_files=cache_files,
                         cache_format=cache_format,
                         cache_save_frequency=cache_save_frequency,
                         max_cache_size=max_cache_size)

        self._host = host
        self._port = port
        if isinstance(indices, str):
            self._indices = indices
        else:
            self._indices = ",".join(indices)
        self._query_format = query_format
        self._max_question_length = max_question_length
        self._max_document_length = max_document_length
        self._num_retrievals = num_retrievals

        self._es = Elasticsearch(hosts=[{"host": host, "port": port}],
                                 retries=retries,
                                 timeout=timeout)


    @overrides
    def _cache_key(self, query: Dict[str, str]):
        query_new = query
        if self._query_format == "aristo-qstem" and 'a' in query:
            query_new = query.copy()
            del query_new['a']
        return super()._cache_key(query_new)

    @overrides
    def _retriever(self, query: Dict[str, str]) -> List[Dict[str, Any]]:
        question = query.get('q')
        answer = query.get('a', "")
        if self._query_format == "aristo-qstem":
            answer = ""
        max_hits = self._num_retrievals
        if self._max_document_length and max_hits:
            max_hits *= 2  # Manual heuristic factor to account for filtered long documents
        es_query = self.construct_qa_query(question,
                                           answer,
                                           require_match=True,
                                           max_hits=max_hits,
                                           max_question_length=self._max_question_length)
        res = self._es.search(index=self._indices, body=es_query)
        hits = res['hits']['hits']
        if self._max_document_length:
            hits = list(filter(lambda x:len(x['_source']['text']) <= self._max_document_length, hits))
        if self._num_retrievals is not None:
            hits = hits[:self._num_retrievals]
        return [{'score': hit['_score'], 'text': hit['_source']['text']} for hit in hits]

    @staticmethod
    def construct_qa_query(question, choice="", require_match=True, max_hits=50, max_question_length=None):
        question_for_query = question[-max_question_length:] if max_question_length else question
        query_text = question_for_query + " " + choice
        if require_match and len(choice) > 0:
            return {"from": 0, "size": max_hits,
                    "query": {
                        "bool": {
                            "must": [
                                {"match": {
                                    "text": query_text
                                }}
                            ],
                            "filter": [
                                {"match": {"text": choice}},
                                {"match": {"text": question_for_query}},
                                {"type": {"value": "sentence"}}
                            ]
                        }
                    }}
        return {"from": 0, "size": max_hits,
                "query": {
                    "bool": {
                        "must": [
                            {"match": {
                                "text": query_text
                            }}
                        ],
                        "filter": [
                            {"type": {"value": "sentence"}}
                        ]
                    }
                }}