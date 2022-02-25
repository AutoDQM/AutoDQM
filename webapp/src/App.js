import React, {Component} from 'react';
import {
  BrowserRouter as Router,
  Switch,
  Route,
  NavLink as RouterNavLink,
} from 'react-router-dom';
import {Nav, NavItem, NavLink} from 'reactstrap';
import InputPage from './input/InputPage.js';
import PlotsPage from './plots/PlotsPage.js';
import * as api from './api.js';

class App extends Component {
  constructor(props) {
    super(props);
    const recentQueryString = localStorage.getItem('recentQuery');
    const recentQuery = recentQueryString && JSON.parse(recentQueryString);
    this.state = {
      recentQuery: recentQuery,
    };
  }

  handleNewQuery = query => {
    localStorage.setItem('recentQuery', JSON.stringify(query));
    this.setState({recentQuery: query});
  };

  render() {
    const plotsUrl = this.state.recentQuery
      ? api.queryUrl(this.state.recentQuery)
      : '';
    return (
      <Router basename="/dqm/autodqm/">
        <React.Fragment>
          <Nav tabs>
            <NavItem>
              <NavLink
                tag={RouterNavLink}
                exact={true}
                to="/"
                activeClassName="active">
                AutoDQM
              </NavLink>
            </NavItem>
            <NavItem>
              {plotsUrl && (
                <NavLink
                  tag={RouterNavLink}
                  to={plotsUrl}
                  activeClassName="active">
                  Plots
                </NavLink>
              )}
            </NavItem>
          </Nav>
          <Switch>
            <Route
              exact
              path="/"
              render={props => (
                <InputPage recentQuery={this.state.recentQuery} {...props} />
              )}
            />
            <Route
              path="/plots/:subsystem/:refSeries/:refSample/:refRun/:dataSeries/:dataSample/:dataRun"
              render={props => (
                <PlotsPage onNewQuery={this.handleNewQuery} {...props} />
              )}
            />
            <Route path="/plots" component={PlotsPage} />
          </Switch>
        </React.Fragment>
      </Router>
    );
  }
}

export default App;
