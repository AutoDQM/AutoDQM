import React, {Component} from 'react';
import {Card, CardBody, Input, Button, Col, Row} from 'reactstrap';
import Switch from 'react-switch';
import {css} from 'react-emotion';
import {Redirect} from 'react-router-dom';
import ApiSelect from '../input/ApiSelect.js';

export default class Controls extends Component {
  handleSearchChange = e => {
    this.props.onSearchChange(e.target.value);
    e.preventDefault();
  };

  render() {
    return (
      <Card className={this.props.className}>
        <CardBody className="p-2">
          <div>
            <small className="text-muted">Filter plots</small>
            <Input
              onChange={this.handleSearchChange}
              value={this.props.search}
            />
          </div>

          <div className="mt-2">
            <small className="text-muted mt-3">
              Select a different data run
            </small>
            <Row>
              <RunSwitch {...this.props.query} />
            </Row>
          </div>

          <div className="mt-3">
            <span>Show hidden plots</span>
            <Switch
              className={css`
                vertical-align: middle;
                margin-left: 4px;
                float: right;
              `}
              checked={this.props.showAll}
              onChange={this.props.onShowAllChange}
            />
          </div>
        </CardBody>
      </Card>
    );
  }
}

class RunSwitch extends Component {
  constructor(props) {
    super(props);
    this.state = {
      runOpts: [],
      delay: null,
      redirect: null,
    };
  }

  componentWillUnmount = () => {
    this.setState({delay: null, redirect: null});
  };

  componentDidUpdate = prevProps => {
    if (this.state.redirect) this.setState({redirect: null});
    if (
      prevProps.dataSeries !== this.props.dataSeries ||
      prevProps.dataSample !== this.props.dataSample
    ) {
      this.setState({runOpts: []});
    }
  };

  switchRun = run => {
    const params = [
      this.props.dqmSource,
      this.props.subsystem,
      this.props.refSeries,
      this.props.refSample,
      this.props.refRun,
      this.props.dataSeries,
      this.props.dataSample,
      run,
    ];
    const redirect = `/plots/${params.join('/')}`;
    const delay = setTimeout(this.endDelay, this.props.delayDuration || 3000);
    this.setState({redirect, delay});
  };

  endDelay = () => {
    if (this.state.delay) this.setState({delay: null});
  };

  onRunsLoad = runOpts => {
    runOpts.sort((a,b) => a.label.localeCompare(b.label));
    this.setState({runOpts});
  };

  render() {
    const rOpts = this.state.runOpts;
    const curIdx = rOpts.map(o => o.value).indexOf(this.props.dataRun);
    const prevRun = rOpts[curIdx - 1] && rOpts[curIdx - 1].value;
    const nextRun = rOpts[curIdx + 1] && rOpts[curIdx + 1].value;

    return (
      <React.Fragment>
        {this.state.redirect && <Redirect to={this.state.redirect} />}
        <Col xs="4">
          <SeekButton
            onClick={() => this.switchRun(prevRun)}
            disabled={!prevRun || (this.state.delay && true)}
            title={prevRun && prevRun}>
            Prev
          </SeekButton>
        </Col>
        <Col xs="4" className="p-0">
          <ApiSelect
            type="get_runs"
            series={this.props.dataSeries}
            sample={this.props.dataSample}
            onChange={c => this.switchRun(c.value)}
            onLoad={this.onRunsLoad}
            isDisabled={this.state.delay}
          />
        </Col>
        <Col xs="4">
          <SeekButton
            onClick={() => this.switchRun(nextRun)}
            disabled={!nextRun || (this.state.delay && true)}
            title={nextRun && nextRun}>
            Next
          </SeekButton>
        </Col>
      </React.Fragment>
    );
  }
}

function SeekButton(props) {
  const {children, ...otherProps} = props;
  return (
    <Button
      outline
      color="primary"
      className={
        css`
          width: 100%;
        ` + ' text-truncate'
      }
      {...otherProps}>
      {children}
    </Button>
  );
}
