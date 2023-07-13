import React, {Component} from 'react';
import {Card, CardHeader, CardImg} from 'reactstrap';
import {css, cx} from 'react-emotion';

export default class Plots extends Component {
  render() {
    let plotObjs = this.props.plots;

    const plots = plotObjs.map(p => {
      return (
        // <Plot
        //   key={p.id}
        //   name={p.name}
        //   pngUri={p.png_path}
        //   pdfUri={p.pdf_path}
        //   search={this.props.search}
        //   display={shouldDisplay(p, this.props.showAll, this.props.search)}
        //   onHover={() => this.props.onHover(p)}
        // />

	      // <div style="position:relative;">
	      // <a href={p.png_path} target="_blank">
	      // <div style="position:absolute;  z-index:500;height:245px;width:100%;"></div>
	      // <iframe id="forecast_embed" type="text/html" frameborder="0" height="245" width="100%" src={p.png_path}></iframe>
	      // </a>
	      // </div>
	  
	  //you are  going to have to go back to create a columized div in the plots page 
	      <div style={{position:"relative"}}>
	      <a href={p.png_path} target="_blank">
	      <div style={{position:"absolute",  height:"350px", width:"400px"}}></div>
	      <iframe id={p.id} type="text/html" frameborder="0" height="350px" width="400px" src={p.png_path}></iframe>
	      </a>
	      </div>



	      //<div style={{position:'absolute', height:'350px', width:'400px'}}></div>
	      //<a href={p.png_path} target="_blank">
	      //<iframe id="forecast_embed" type="text/html" frameborder="0" height="350px" width="400px" src={p.png_path}></iframe>
	      //</a>
	      
	      
	  
	      // <div style="position:relative;">
	      // <iframe src={p.png_path} height='300' width='450' />
	      // <a href={p.png_path} style="position:absolute; top:0; left:0; display:inline-block; width:500px; height:500px; z-index:5;" target="_blank"></a>
	      //</div> 
	//<div dangerouslySetInnerHTML={__html: {p.png_path}} />
      );
    });
    return <div className={containerSty}>{plots}</div>;
  }
}

const Plot = ({name, pngUri, pdfUri, search, display, onHover}) => {
    return (
    <Card className={cx(plotSty, display ? null : hidden)} onMouseEnter={onHover}>
      <a href={pdfUri} target="_blank">
        <CardHeader>{hlSearch(name, search)}</CardHeader>
        <CardImg src={pngUri} />
      </a>
    </Card>
  
  );
};

const containerSty = css`
  margin-top: 0.5em;
`;

const hidden = css`
  display: none;
`

const mh = '0.5em';
const plotSty = css`
  width: calc(100% / 1 - 2 * ${mh});
  display: inline-block;
  margin: ${mh};
  @media (min-width: 576px) {
    width: calc(100% / 1 - 2 * ${mh});
  }

  @media (min-width: 768px) {
    width: calc(100% / 2 - 2 * ${mh});
  }

  @media (min-width: 992px) {
    width: calc(100% / 3 - 2 * ${mh});
  }

  @media (min-width: 1200px) {
    width: calc(100% / 4 - 2 * ${mh});
  }

  :hover {
    border-color: #6c757d;
  }
`;

const shouldDisplay = (plot, showAll, search) => {
  if(!plot.display && !showAll) return false;
  if(search && plot.name.indexOf(search) === -1) return false;
  return true;
}

const hlSearch = (text, search) => {
  if (!search) return <span>{text}</span>;
  const len = search.length;
  const idx = text.indexOf(search);
  return (
    <span>
      {text.substring(0, idx)}
      <b>{text.substring(idx, idx + len)}</b>
      {text.substring(idx + len)}
    </span>
  );
};
