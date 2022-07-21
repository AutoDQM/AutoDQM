import React from 'react';
import {
  Card,
  CardBody,
  CardTitle,
  CardText,
  ListGroup,
  ListGroupItem,
} from 'reactstrap';

export default function Preview(props) {
  const plot = props.plot;

  let inner;
  if (plot) {
    inner = (
      <React.Fragment>
        <img
          src={plot.png_path}
          alt="Plot Preview"
          className="card-img-top img-fluid"
        />
        <ListGroup flush>{resultItems(plot)}</ListGroup>
      </React.Fragment>
    );
  } else {
    inner = (
      <CardBody>
        <CardTitle>Preview</CardTitle>
        <CardText>Hover over a plot to show more details.</CardText>
      </CardBody>
    );
  }

  return <Card className={props.className}>{inner}</Card>;
}

const resultItems = plot => {
  const rows = [
    {label: 'Name', value: plot.name},
    {label: 'Comparator', value: plot.comparator},
    {label: 'Anomalous', value: plot.display.toString()},
  ];
  for (let key in plot.results) {
    rows.push({label: key, value: plot.results[key]});
  }
  return rows.map(r => (
    <ListGroupItem
      key={r.label}
      className="justify-content-between d-flex w-100 p-2">
      <small className="text-muted">{r.label}</small>
      <span />
      <span className="text-truncate">{r.value}</span>
    </ListGroupItem>
  ));
};
