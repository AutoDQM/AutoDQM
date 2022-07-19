import React, {Component} from 'react';
import * as api from '../api.js';
import Select from 'react-select';
import {css, cx} from 'react-emotion';

export default class ApiSelect extends Component {
  constructor(props) {
    super(props);
    this.state = {
      req: null,
      opts: [],
      err: null,
      flashing: false,
    };
  }

  componentWillMount = () => {
    this.loadOptions();
  };

  componentWillUnmount = () => {
    this.state.req && this.state.req.cancel();
  };

  componentDidUpdate = prevP => {
    const curP = this.props;
    if (
      prevP.type !== curP.type ||
      prevP.dqmSource !== curP.dqmSource ||
      prevP.subsystem !== curP.subsystem ||
      prevP.series !== curP.series ||
      prevP.sample !== curP.sample ||
      prevP.onError !== curP.onError ||
      prevP.onLoad !== curP.onLoad
    ) {
      this.state.req && this.state.req.cancel();
      this.setState({req: null, opts: [], err: null});
      this.loadOptions();
    }
    if (
      (prevP.value || curP.value) &&
      (!prevP.value || !curP.value || prevP.value.value !== curP.value.value)
    ) {
      this.setState({flashing: true});
    }
  };

  handleFlashEnd = () => {
    this.setState({flashing: false});
  };

  loadOptions = () => {

    const {type, dqmSource, subsystem, series, sample, onError, onLoad} = this.props;
    let req;
    if (type === 'get_runs' && dqmSource && subsystem && series && sample) {
      req = api.getRuns(dqmSource, subsystem, series, sample);
    } else if (type === 'get_samples' && dqmSource && series) {
      req = api.getSamples(dqmSource, series);
    } else if (type === 'get_series' && dqmSource) {
      req = api.getSeries(dqmSource);
    } else if (type === 'get_subsystems') {
      req = api.getSubsystems();
    } else if (type === 'get_dqmSources') {
      req = api.getDqmSources();
    } else {
      return;
    }
    this.setState({req});

    req
      .then(res => {
        const seen = {};
        const opts = res.items
          .filter(r => !seen.hasOwnProperty(r.name) && (seen[r.name] = true))
          .map(r => ({value: r.name, label: r.name}))
          .sort((a, b) => a.label.localeCompare(b.label));
        onLoad && onLoad([...opts]);

        if ( type === 'get_runs' ||
	     (dqmSource === "Online" && (type === 'get_series' || type === 'get_samples')) ) opts.reverse();
        this.setState({opts, req: null});
      })
      .catch(err => {
        onError && onError(err);
        if (!(err.type === 'cancel')) this.setState({req: null, err: err});
      });
  };

  render() {
    const {type, dqmSource, subsystem, series, sample, onError, onLoad, ...selectProps} = this.props;
    return (
      <div style={{position: "relative"}}>
        <Select
          options={this.state.opts}
          isLoading={this.state.req}
          className={this.state.err ? errSty : ''}
          {...selectProps}
        />
        <div
          onAnimationEnd={this.handleFlashEnd}
          className={cx(this.state.flashing && flashingSty, overlaySty)}>
        </div>
      </div>
    );
  }
}

const errSty = css`
  > div {
    border-color: rgb(220, 53, 69);
  }
`;

const overlaySty = css`
  position: absolute;
  pointer-events: none;
  top: 0;
  left: 0;
  height: 100%;
  width: 100%;
`;

const flashingSty = css`
  animation: flash 0.5s;

  @keyframes flash {
    from {
      background-color: rgba(23, 162, 184, 0.5);
    }
    to {
      background-color: rgba(23, 162, 184, 0);
    }
  }
`;
