import React, {Component} from 'react';
import ApiSelect from './ApiSelect.js';
import {Row, Col, Button} from 'reactstrap';
import {css} from 'react-emotion';
import {Link} from 'react-router-dom';
import * as api from '../api.js';

export default class InputForm extends Component {
  queryUrl = () => {
    const query = this.props.query;
    return api.queryUrl(query);
  };

  handleDqmSourceChange = change => {
    this.props.onChange({dqmSource: change.value});
  };

  handleSubsystemChange = change => {
    this.props.onChange({subsystem: change.value});
  };

  handleDataChange = (type, change) => {
    if (type === 'series') this.props.onChange({dataSeries: change.value});
    else if (type === 'sample') this.props.onChange({dataSample: change.value});
    else if (type === 'run') this.props.onChange({dataRun: change.value});
  };

  handleRefChange = (type, change) => {
    if (type === 'series') this.props.onChange({refSeries: change.value});
    else if (type === 'sample') this.props.onChange({refSample: change.value});
    else if (type === 'run') this.props.onChange({refRun: change.value});
  };

  render() {
    const q = this.props.query;
    const data = {dqmSource: q.dqmSource, subsystem: q.subsystem, series: q.dataSeries, sample: q.dataSample, run: q.dataRun};
    const ref = {dqmSource: q.dqmSource, subsystem: q.subsystem, series: q.refSeries, sample: q.refSample, run: q.refRun};

    return (
      <React.Fragment>
        <Row className="mt-3">
          <Col>
            <Row>
              <Col>
                <h3>Source</h3>
              </Col>
              <Col/>
            </Row>
            <ApiSelect
              placeholder="Select DQM source..."
              type="get_dqmSources"
              value={option(q.dqmSource)}
              onChange={this.handleDqmSourceChange}
            />
          </Col>
          <Col>
            <Row>
              <Col>
                <h3>Subsystem</h3>
              </Col>
              <Col/>
              <Col xs="auto">
                <Button onClick={this.props.onClearForm} color="link">
                  clear form
                </Button>
              </Col>
            </Row>
            <ApiSelect
              placeholder="Select subsystem..."
              type="get_subsystems"
              value={option(q.subsystem)}
              onChange={this.handleSubsystemChange}
            />
          </Col>
        </Row>
        <Row className="mt-3">
          <Col md="6">
            <h3>Data Run</h3>
            <RunSelectForm {...data} onChange={this.handleDataChange} />
          </Col>
          <Col md="6">
            <h3>Reference Run</h3>
            <RunSelectForm {...ref} onChange={this.handleRefChange} />
          </Col>
        </Row>
        <Row className="mt-3">
          <Col>
            <Button
              color="success"
              disabled={!this.queryUrl()}
              className={css`
                width: 100%;
                margin-top: 10px;
              `}
              tag={Link}
              to={this.queryUrl() || ''}>
              Submit
            </Button>
          </Col>
        </Row>
      </React.Fragment>
    );
  }
}

function RunSelectForm(props) {
  return (
    <React.Fragment>
      <span>Series</span>
      <ApiSelect
        placeholder="Select series..."
        type="get_series"
        dqmSource={props.dqmSource}
        value={option(props.series)}
        onChange={c => props.onChange('series', c)}
      />
      <span>Sample</span>
      <ApiSelect
        placeholder="Select sample..."
        type="get_samples"
        dqmSource={props.dqmSource}
        series={props.series}
        value={option(props.sample)}
        onChange={c => props.onChange('sample', c)}
      />
      <span>Run</span>
      <ApiSelect
        placeholder="Select run..."
        type="get_runs"
        dqmSource={props.dqmSource}
        subsystem={props.subsystem}
        series={props.series}
        sample={props.sample}
        value={option(props.run)}
        onChange={c => props.onChange('run', c)}
      />
    </React.Fragment>
  );
}

function option(val) {
  if (val === null || val === undefined) return null;
  return {value: val, label: val};
}
