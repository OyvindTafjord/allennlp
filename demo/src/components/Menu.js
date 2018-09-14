import React from 'react';
import { Link } from 'react-router-dom';

/*******************************************************************************
  <Header /> Component
*******************************************************************************/

class Menu extends React.Component {
    render() {
      const { selectedModel, clearData } = this.props;

      const buildLink = (thisModel, label) => {
        return (
          <li>
            <span className={`nav__link ${selectedModel === thisModel ? "nav__link--selected" : ""}`}>
              <Link to={"/" + thisModel} onClick={clearData}>
                <span>{label}</span>
              </Link>
            </span>
          </li>
        )
      }

      return (
        <div className="menu">
          <div className="menu__content">
            <h1 className="menu__content__logo">
              <a href="http://www.allennlp.org/" target="_blank" rel="noopener noreferrer">
                <svg>
                  <use xlinkHref="#icon__allennlp-logo"></use>
                </svg>
                <span className="u-hidden">AllenNLP</span>
              </a>
            </h1>
            <nav>
              <ul>
                {buildLink("quarel-q-parsing", "QuaRel QA (QuaSP+)")}
              </ul>
              <ul>
                {buildLink("friction-q-parsing", "Friction QA (QuaSP+)")}
              </ul>
              <ul>
                {buildLink("quarel-q-parsing-zero", "QuaRel QA (Zero shot)")}
              </ul>
              <ul>
                {buildLink("friction-q-parsing-zero", "Friction QA (Zero shot)")}
              </ul>
            </nav>
          </div>
        </div>
      );
    }
  }

export default Menu;
