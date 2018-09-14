import React from 'react';
import HeatMap from './heatmap/HeatMap'
import Collapsible from 'react-collapsible'
import { API_ROOT } from '../api-config';
import { withRouter } from 'react-router-dom';
import {PaneLeft, PaneRight} from './Pane'
import Button from './Button'
import ModelIntro from './ModelIntro'


/*******************************************************************************
 <McInput /> Component
 *******************************************************************************/

var parserExamples = {};
parserExamples["friction"] = [
  {
    question: "William is ice skating and notices that his ice skates glides quicker on wet ice as opposed to freshly fallen snow. The reason for this is because there is more friction on the (A) wet ice (B) freshly fallen snow"
  },
  {
    question: "Joe went sledding and found that using the other sledders paths, he could go much faster This is because the path was _____ than the untouched snow? (A) smoother (B) rougher"
  },
  {
    question: "Johnny has a snowmobile that he rides over the snow and rocky road. He noticed that he moves faster on the snow because it has (A) more resistance (B) less resistance"
  },
  {
    question: "Lydia pushes a kumquat across a silk sheet on one bed and a cotton sheet on another bed. The kumquat has less friction when Lydia pushes it across the silk sheet than when she pushes it across the cotton sheet. When she pushes it across the cotton sheet, the distance it goes compared to the distance on the silk sheet will be (A) larger (B) smaller"
  },
  {
    question: "I tried a fine, smoother emery board that filed my nail more slowly and gently while rough one did it more quickly with more damage. This is because the rougher board caused _____friction. (A) more (B) less"
  },
  {
    question: "Bryan enjoys playing basketball with his young daughter. After shooting some hoops, they sit down and roll the ball to each other. Sometimes Bryan takes her to his college gym and sometimes he goes to the courts at the park around the corner from his home. He notices that when they play roll-the-ball, the ball goes faster on the wooden gym floor than on the concrete basketball court at the park. Which surface has more friction? (A) the wooden gym floor (B) the concrete"
  },
  {
    question: "Rebecca rolls a grape across a wooden table and across a hay bale. The hay bale is rougher than the wooden table. The surface with the most resistance is (A) wooden table (B) hay bale"
  },
  {
    question: "Gina rolls a cherry on one plate covered in water and another plate covered in honey. The cherry moves slower on the honey plate. The cherry generates less heat on the (A) water plate (B) honey plate"
  },
  {
    question: "Emily ran her hand through her hair, and then over the skin on her arm. She noticed that her skin felt smoother, which means that the one with less friction is her (A) hair (B) skin."
  },
  {
    question: "Tom twisted a piece of wire around and around until the piece snapped off, which he noticed left the ends much hotter than when he just snipped the wire. The twisting repeatedly caused _____ then snipping? (A) more friction (B) less friction"
  },
  {
    question: "Dan thought that his new hockey buck would go faster off the ice and on a court. So he decided to conduct a test and he determined that he puck went faster on (A) the ice or (B) the basketball court."
  },
  {
    question: "Mary was rolling a log down a hill. She noticed that when she rolls the log over the rocky part of the hill that is very bumpy, the log struggles to roll compared to when she rolls it down the muddy part of the hill. The section of the hill with the most resistance is the (A) rocky part of the hill (B) muddy part of the hill."
  },
  {
    question: "Timmy is rolling a toy school back and forth over a couple of different surfaces. He noticed that when he rolls the school bus over a smooth sheet of steal it creates less friction than if he were to roll it over a sheet of sandpaper. Will the toy school bus generate less heat when it's rolled on the (A) Sheet of steal (B) Sandpaper"
  },
  {
    question: "Billy rubbed a slab of raw chicken back and forth across two different surfaces. One of the surfaces that he rubbed the raw chicken on was a wooden table top, and the other surface was a square foot of velcro. Billy noticed that the slab of raw chicken seemed to be warmer after rubbing it on velcro as opposed to when he rubbed it on the wooden table top. Billy concluded that the rougher of the two surfaces was the (A) wooden table top (B) velcro"
  },
  {
    question: "A skating rink has less friction than a parking lot. Riding on roller skates in _____ allows you to travel further (A) skating rink (B) parking lot"
  },
  {
    question: "A bee flew quicker through clear skies than wind because the wind has much (A) lower resistance (B) higher resistance"
  },
  {
    question: "Jenny dropped a spool of thread and watched it roll rather quickly across the slick hardwood floor. She picked up the spool, then dropped it on the the textured rubber mat in the kitchen, which was rather rougher than the wood floor. The spool moved noticeably slower. Which of the surfaces had the greatest amount of friction? (A) hardwood floor (B) textured rubber mat"
  },
  {
    question: "Tim is trying out a new skate park. The skate park has smooth ramps and he likes it much better than the cracked parking lot that he used to skate on. There will be less resistance on the (A) new skate park (B) old parking lot."
  },
  {
    question: "Casey's auger got a lot hotter when used in clay compared to when he used it in dirt. The augur traveled further in the (A) dirt (B) clay"
  },
  {
    question: "James discovered that rolling his yo-yo over a porcelain tile floor generated less heat than rolling it over his doormat. James' yo-yo moved slower over (A) porcelain tile (B) doormat"
  }
];

parserExamples["quarel"] = [
  {
    question: "Mary has always been weaker then Jimbo. Which person is able to throw a ball farther? (A) Jimbo (B) Mary"
  },
  {
    question: "An empty pot generates less heat when Mary slides it the wood counter than it does when she slides it across a dish towel. This is because the _____ has less resistance. (A) towel (B) wood counter"
  },
  {
    question: "A hockey puck slides a lot longer on a frozen lake then on a slushy lake. This means the surface of the _____ is smoother (A) frozen lake (B) slushy lake"
  },
  {
    question: "The sun has much more mass than the earth so it has (A) weaker gravity (B) stronger gravity"
  },
  {
    question: "Annabel and Lydia are riding their tricycles. Lydia is not as fast as Annabel is. Which one goes a shorter distance? (A) Annabel (B) Lydia"
  },
  {
    question: "Connie's toy soldier slides a lot slower over the hallway floor then over the bedroom floor. She figures this makes the _____ much rougher (A) bedroom floor (B) hallway floor"
  },
  {
    question: "Miller thinks that a light that he is approaching is very small, since he is quite far away. As he gets closer, he sees that it is a huge spotlight. The light is actually (A) bright (B) dim"
  },
  {
    question: "Kory set up two slides in his backyard. One slide is metal and doesn't create much friction, while the other slide is wood and creates more friction. If Kory slides down both slides, which one will he go quicker on? (A) metal slide (B) wood slide"
  },
  {
    question: "Michael and Daphne leave the train station at the same time and begin walking home along the same route. Michael decides to stop at a friend's house along the way, while Daphne continues walking. Who would end up going the smaller distance from the train station? (A) Michael (B) Daphne"
  },
  {
    question: "The fastest land animal on earth, a cheetah was having a 100m race against a rabbit. Which one one the race? (A) the cheetah (B) the rabbit"
  },
  {
    question: "Jim was playing with his new ball. He rolled it on the carpet in his living room and it didn't go very far. He decided to go outside and play. Jim rolled his ball on the concrete driveway and it went a much longer distance. He realized there was more resistance to the ball on (A) the carpet (B) the concrete."
  },
  {
    question: "If Mona is doing pushups and Milo is reading a book, which person is sweating less? (A) Mona (B) Milo"
  },
  {
    question: "James has rigorous fencing lessons in the morning, and in the evening goes to see a film at the cinema with his girlfriend. Where is he less likely to be sweaty? (A) While fencing (B) While watching a movie"
  },
  {
    question: "Jose pushed his burrito cart on the bumpy sidewalk and went slow, while much faster on the street because it had (A) more resistance (B) less resistance"
  },
];

var title = {};
var description = {};

title["friction"] = "Friction Story Question Answering";
title["quarel"] = "Qualitative Relations Story Question Answering";
description["friction"] = (
  <span>
    Answer story questions about friction (QuaRel<sup>F</sup> dataset). This uses the QuaSP+
    semantic parser described in <i>QuaRel: A Dataset and Models for
    Answering Questions about Qualitative Relationships</i> (AAAI 2018 submission). The examples
    are from the validation set.
  </span>
);

description["quarel"] = (
  <span>
    Answer story questions about qualitative relations (QuaRel dataset). This uses the QuaSP+
    semantic parser described in <i>QuaRel: A Dataset and Models for
    Answering Questions about Qualitative Relationships</i> (AAAI 2018 submission). The examples
    are from the validation set.
  </span>
);


class FrictionQInput extends React.Component {
  constructor(props) {
    super(props);

    // If we're showing a permalinked result,
    // we'll get passed in a question.
    const { question, mode } = props;

    this.mode = mode;

    this.state = {
      questionValue: question || ""
    };
    this.handleListChange = this.handleListChange.bind(this);
    this.handleQuestionChange = this.handleQuestionChange.bind(this);
  }

  handleListChange(e) {
    if (e.target.value !== "") {
      this.setState({
        questionValue: parserExamples[this.mode][e.target.value].question,
      });
    }
  }

  handleQuestionChange(e) {
    this.setState({
      questionValue: e.target.value,
    });
  }

  render() {

    const { questionValue } = this.state;
    const { outputState, runParser } = this.props;

    const parserInputs = {
      "questionValue": questionValue
    };

    return (
      <div className="model__content">
        <ModelIntro title={title[this.mode]} description={description[this.mode]} />
        <div className="form__instructions"><span>Enter text or</span>
          <select disabled={outputState === "working"} onChange={this.handleListChange}>
            <option value="">Choose an example...</option>
            {parserExamples[this.mode].map((example, index) => {
              return (
                <option value={index} key={index}>{example.question.substring(0,60) + "..."}</option>
              );
            })}
          </select>
        </div>
        <div className="form__field">
          <label htmlFor="#input--mc-question">Question</label>
          <textarea onChange={this.handleQuestionChange} id="input--mc-question" type="text" required="true" value={questionValue} placeholder="E.g. &quot;William is ice skating and notices that his ice skates glides quicker on wet ice as opposed to freshly fallen snow. The reason for this is because there is more friction on the (A) wet ice (B) freshly fallen snow&quot;" disabled={outputState === "working"} />
        </div>
        <div className="form__field form__field--btn">
          <Button enabled={outputState !== "working"} runModel={runParser} inputs={parserInputs} />
        </div>
      </div>
    );
  }
}


/*******************************************************************************
 <FrictionQOutput /> Component
 *******************************************************************************/

class FrictionQOutput extends React.Component {
  render() {
    const { answer, logicalForm, score, actions, question_tokens, worldExtractions, explanation } = this.props;

    return (
      <div className="model__content">
        <div className="form__field">
          <label>Answer</label>
          <div className="model__content__summary">{ answer }</div>
        </div>

        <div className="form__field">
          <label>Explanation</label>
          <div className="model__content__summary">
            {explanation.map((entry) => (
              <div><div className="friction__explanation__header">{ entry.header }:</div>
                <ul className="friction__explanation__ul">{
                  entry.content.map((c) => (<li className="friction__explanation__ul">{c}</li>))
                }</ul>
              </div>
            ))}
          </div>
        </div>

        <div className="form__field">
          <Collapsible trigger="Model internals">
            <div className="form__field">
              <label>Logical Form</label>
              <div className="model__content__summary">{ logicalForm }</div>
            </div>
            <div className="form__field">
              <label>Score</label>
              <div className="model__content__summary">{ score }</div>
            </div>

            <div className="form__field">
              <label>Extracted world entities</label>
              <div className="model__content__summary">world1: { worldExtractions.world1 },
                world2: { worldExtractions.world2 }</div>
            </div>
            <Collapsible trigger="Predicted actions">
              {actions.map((action, action_index) => (
                <Collapsible key={"action_" + action_index} trigger={action['predicted_action']}>
                  <ActionInfo action={action} question_tokens={question_tokens}/>
                </Collapsible>
              ))}
            </Collapsible>
          </Collapsible>
        </div>
      </div>
    );
  }
}


class ActionInfo extends React.Component {
  render() {
    const { action, question_tokens } = this.props;
    const question_attention = action['question_attention'].map(x => [x]);
    const considered_actions = action['considered_actions'];
    const action_probs = action['action_probabilities'].map(x => [x]);

    const probability_heatmap = (
      <div className="heatmap">
        <HeatMap xLabels={['Prob']} yLabels={considered_actions} data={action_probs} xLabelWidth={250} />
      </div>
    );
    const question_attention_heatmap = question_attention.length > 0 ? (
      <div className="heatmap">
        <HeatMap xLabels={['Prob']} yLabels={question_tokens} data={question_attention} xLabelWidth={70} />
      </div>
    ) : (
      ""
    )


    return (
      <div>
        {probability_heatmap}
        {question_attention_heatmap}
      </div>
    )
  }
}


/*******************************************************************************
 <McComponent /> Component
 *******************************************************************************/

class _FrictionQParserComponent extends React.Component {

  constructor(props) {
    super(props);

    const { requestData, responseData, mode } = props;

    this.mode = mode;

    this.state = {
      outputState: responseData ? "received" : "empty", // valid values: "working", "empty", "received", "error"
      requestData: requestData,
      responseData: responseData
    };

    this.runParser = this.runParser.bind(this);
  }

  runParser(event, inputs) {
    this.setState({outputState: "working"});

    const urlStub = this.mode + "-q-parsing"

    var payload = {
      question: inputs.questionValue,
    };
    fetch(`${API_ROOT}/predict/${urlStub}`, {
      method: 'POST',
      headers: {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload)
    }).then((response) => {
      return response.json();
    }).then((json) => {
      // If the response contains a `slug` for a permalink, we want to redirect
      // to the corresponding path using `history.push`.
      const { slug } = json;
      const newPath = slug ? `/${urlStub}/` + slug : `/${urlStub}`;

      // We'll pass the request and response data along as part of the location object
      // so that the `Demo` component can use them to re-render.
      const location = {
        pathname: newPath,
        state: { requestData: payload, responseData: json }
      }
      this.props.history.push(location);
    }).catch((error) => {
      this.setState({outputState: "error"});
      console.error(error);
    });
  }

  render() {
    const { requestData, responseData, mode } = this.props;

    const question = requestData && requestData.question;
    const answer = responseData && responseData.answer;
    const logicalForm = responseData && responseData.logical_form;
    const score = responseData && responseData.score;
    const actions = responseData && responseData.predicted_actions;
    const question_tokens = responseData && responseData.question_tokens;
    const worldExtractions = responseData && responseData.world_extractions;
    const explanation = responseData && responseData.explanation;

    return (
      <div className="pane model">
        <PaneLeft>
          <FrictionQInput runParser={this.runParser}
                          outputState={this.state.outputState}
                          mode={mode}
                          question={question}/>
        </PaneLeft>
        <PaneRight outputState={this.state.outputState}>
          <FrictionQOutput answer={answer}
                            logicalForm={logicalForm}
                            score={score}
                            actions={actions}
                            question_tokens={question_tokens}
                            worldExtractions={worldExtractions}
                            explanation={explanation}
          />
        </PaneRight>
      </div>
    );

  }
}

const FrictionQParserComponent = withRouter(_FrictionQParserComponent);

export default FrictionQParserComponent;