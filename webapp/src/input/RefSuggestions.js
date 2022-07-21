import React, {Component} from 'react';
import {
  Row,
  Col,
  Alert,
  Card,
  CardHeader,
  CardBody,
  ListGroup,
  ListGroupItem,
  Progress,
} from 'reactstrap';
import * as api from '../api.js';

export default class RefSuggestions extends Component {
  state = {
    refReq: null,
    refCands: [],
    error: null,
  };

  componentWillMount = () => {
    this.loadRefs();
  };

  componentWillUnmount = () => {
    this.state.refReq && this.state.refReq.cancel();
  };

  componentDidUpdate = prevProps => {
    if (
      this.props.dqmSource !== prevProps.dqmSource ||
      this.props.subsystem !== prevProps.subsystem ||
      this.props.series !== prevProps.series ||
      this.props.sample !== prevProps.sample ||
      this.props.run !== prevProps.run
    ) {
      this.loadRefs();
    }
  };

  loadRefs = () => {
    let {dqmSource, subsystem, series, sample, run} = this.props;

    if (!dqmSource || !subsystem || !series || !sample || !run) {
      this.state.refReq && this.state.refReq.cancel();
      this.setState({refReq: null, refCands: []});
    } else {
      let r = api.getReferences(dqmSource, subsystem, series, sample, run);
      this.setState({refReq: r});

      r.then(res => {
        this.setState({refReq: null, refCands: res.candidates});
      }).catch(err => {
        if (!(err.type === 'cancel'))
          this.setState({refReq: null, error: err, refCands: []});
      });
    }
  };

  renderList = () => {
    let {series, sample} = this.props;
    return (
      <ListGroup>
        {this.state.refCands.map(r => (
          <RefItem
            key={r.run}
            {...r}
            onClick={() => this.props.onChange({...r, series, sample})}
          />
        ))}
      </ListGroup>
    );
  };

  render() {
    let body;
    if (this.state.refReq) {
      body = (
        <CardBody>
          <Row style={{alignItems: 'center'}}>
            <Col xs="auto" className="text-muted">
              Loading reference suggestions...{' '}
            </Col>
            <Col>
              <Progress animated={true} color="info" value={100} />
            </Col>
          </Row>
        </CardBody>
      );
    } else if (this.state.error) {
      body = (
        <Alert color="danger" className="mb-0">
          References could not be loaded: {this.state.error.message}
        </Alert>
      );
    } else if (this.state.refCands.length === 0) {
      body = (
        <CardBody>Select a data run to load potential reference runs</CardBody>
      );
    } else {
      body = this.renderList();
    }

    return (
      <Card className="my-4">
        <CardHeader>Reference Run Suggestions</CardHeader>
        {body}
      </Card>
    );
  }
}

function RefItem({run, best, run_age, lumi_ratio, onClick}) {
  return (
    <ListGroupItem
      tag="button"
      className="text-left"
      color={best ? 'success' : ''}
      style={{cursor: 'pointer'}}
      onClick={onClick}>
      <div>Run: {run}</div>
      <div>
        Started {run_age.days} days, {run_age.hours} hours, {run_age.minutes}{' '}
        minutes prior
      </div>
      <div>Luminosity Ratio: {lumi_ratio}</div>
    </ListGroupItem>
  );
}
