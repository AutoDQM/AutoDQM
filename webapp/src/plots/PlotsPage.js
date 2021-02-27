import React, {Component} from 'react';
import {css} from 'react-emotion';
import {
  Card,
  CardTitle,
  CardText,
  Button,
  Container,
  Row,
  Col,
  Progress,
} from 'reactstrap';
import Controls from './Controls.js';
import Preview from './Preview.js';
import ReportInfo from './ReportInfo.js';
import Plots from './Plots.js';
import {Link} from 'react-router-dom';
import * as api from '../api.js';

const fullHeight = css`
  height: 100%;
  overflow-y: auto;
`;

export default class PlotsPage extends Component {
  constructor(props) {
    super(props);
    this.state = {
      plots: [],
      error: null,
      refReq: null,
      dataReq: null,
      procReq: null,
      showLoading: false,
      search: '',
      showAll: false,
      hoveredPlot: null,
    };
  }

  componentWillMount = () => {
    this.update();
  };

  componentWillUnmount = () => {
    this.state.refReq && this.state.refReq.cancel();
    this.state.dataReq && this.state.dataReq.cancel();
    this.state.procReq && this.state.procReq.cancel();
  };

  componentDidUpdate = prevProps => {
    if (prevProps.match.url === this.props.match.url) return;
    this.update();
  };

  update = () => {
    if (!this.validParams(this.props.match.params)) {
      this.setState({error: {message: 'Invalid report parameters!'}});
    } else {
      this.props.onNewQuery(this.props.match.params);
      this.loadReport(this.props.match.params);
    }
  };

  handleShowAllChange = checked => {
    this.setState({showAll: checked});
  };

  handleSearchChange = search => {
    this.setState({search});
  };

  handleHover = hoveredPlot => {
    this.setState({hoveredPlot});
  };

  processChunk = ({query, plots, chunk_index, chunk_size}) => {
        const procReq = api.generateReport(query, chunk_index, chunk_size);
        this.setState({refReq: null, dataReq: null, procReq});
        procReq.then(res => {
            plots = plots.concat(res.items);
            chunk_index = res.chunk_index;
            if(chunk_index == -1)
            {
              this.setState({plots, procReq: null, showLoading: false});
            }else
            {
              this.setState({plots});
              this.processChunk({query, plots, chunk_index, chunk_size});
            }
        });
  };

  loadReport = query => {
    const refReq = api.loadRun(query.refSeries, query.refSample, query.refRun);
    const dataReq = api.loadRun(
      query.dataSeries,
      query.dataSample,
      query.dataRun,
    );
    this.setState({refReq, dataReq, showLoading: true});

    refReq.then(res => {
      this.state.refReq && this.setState({refReq: null});
      return res;
    });
    dataReq.then(res => {
      this.state.dataReq && this.setState({refReq: null});
      this.setState({dataReq: null});
      return res;
    });

    var plots = [];
    var chunk_index = 0;
    const chunk_size = 200;

    Promise.all([refReq, dataReq])
      .then(res => {
        this.processChunk({query, plots, chunk_index, chunk_size});
      })
      .catch(err => {
        if (err.type === 'cancel') return;
        this.setState({
          refReq: null,
          dataReq: null,
          error: err,
          showLoading: false,
        });
    });

  };


  validParams = params => {
    return (
      params.subsystem &&
      params.refSeries &&
      params.refSample &&
      params.refRun &&
      params.dataSeries &&
      params.dataSample &&
      params.dataRun
    );
  };

  render() {
    const {refReq, dataReq, procReq, showLoading} = this.state;
    let body;
    if (this.state.error) {
      body = (
        <Card
          body
          outline
          color="danger"
          className="text-center mx-auto mt-3 col-lg-5">
          <CardTitle>Something went wrong...</CardTitle>
          <CardText>{this.state.error.message}</CardText>
          <Button color="primary" outline tag={Link} to="/">
            Return to input page.
          </Button>
        </Card>
      );
    } else if (showLoading) {
      body = (
        <LoadingBox
          refLoading={refReq}
          dataLoading={dataReq}
          procLoading={procReq}
          procStandby={!procReq}
        />
      );
    } else {
      body = (
        <Plots
          plots={this.state.plots}
          search={this.state.search}
          showAll={this.state.showAll}
          onHover={this.handleHover}
        />
      );
    }

    return (
      <Container
        fluid
        className={css`
          flex-grow: 1;
          min-height: 0;
          height: 100%;
        `}>
        <Row
          className={css`
            padding: 0;
            height: 100%;
          `}>
          <Col
            md={4}
            xl={3}
            className={`${fullHeight} d-none d-md-block bg-light p-3`}>
            <Controls
              query={this.props.match.params}
              onShowAllChange={this.handleShowAllChange}
              onSearchChange={this.handleSearchChange}
              showAll={this.state.showAll}
              search={this.state.search}
              onHover={this.handleHover}
            />
            <Preview className="my-3" plot={this.state.hoveredPlot} />
          </Col>
          <Col md={8} xl={9} className={fullHeight}>
            <ReportInfo
              {...this.props.match.params}
              timestamp={new Date().toUTCString()}
            />
            {body}
          </Col>
        </Row>
      </Container>
    );
  }
}

const LoadingBox = ({refLoading, dataLoading, procLoading, procStandby}) => {
  return (
    <Card body outline className="mx-auto mt-3 col-lg-5">
      <CardTitle className="text-center">Loading...</CardTitle>
      <Progress
        animated={!!refLoading}
        color={refLoading ? 'info' : 'success'}
        value={100}
        className="mt-2">
        {refLoading ? 'Reference loading...' : 'Reference loaded!'}
      </Progress>
      <Progress
        animated={!!dataLoading}
        color={dataLoading ? 'info' : 'success'}
        value={100}
        className="mt-2">
        {dataLoading ? 'Data loading...' : 'Data loaded!'}
      </Progress>
      <Progress
        animated={!procStandby && !!procLoading}
        color={procLoading ? 'info' : procStandby ? 'secondary' : 'success'}
        value={100}
        className="mt-2">
        {procLoading
          ? 'Processing...'
          : procStandby
            ? 'Waiting to process...'
            : 'Processed!'}
      </Progress>
    </Card>
  );
};
