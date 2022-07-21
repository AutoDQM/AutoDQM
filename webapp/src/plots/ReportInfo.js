import React from 'react';
import {Card, Container, Row, Col, Table} from 'reactstrap';

export default function ReportInfo(props) {
  return (
    <Container>
      <Card body outline color="info" className="mt-3 p-3">
        <Row>
          <Col md="6" lg="4">
            <h2>AutoDQM Report</h2>
            <h4>DQM source: {props.dqmSource}</h4>
            <h4>Subsystem: {props.subsystem}</h4>
            <small className="text-muted">{props.timestamp}</small>
          </Col>
          <Col md="6" lg="8">
            <Table striped size="sm" className="mb-0">
              <thead className="thead-dark">
                <tr>
                  <th>Details</th>
                  <th>Data Run</th>
                  <th>Ref Run</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <th scope="row">Series</th>
                  <td>{props.dataSeries}</td>
                  <td>{props.refSeries}</td>
                </tr>
                <tr>
                  <th scope="row">Sample</th>
                  <td>{props.dataSample}</td>
                  <td>{props.refSample}</td>
                </tr>
                <tr>
                  <th scope="row">Run</th>
                  <td>{props.dataRun}</td>
                  <td>{props.refRun}</td>
                </tr>
              </tbody>
            </Table>
          </Col>
        </Row>
      </Card>
    </Container>
  );
}
