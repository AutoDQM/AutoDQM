import axios from 'axios';

const API = '/dqm/autodqm/cgi-bin/index.py';

export function getDqmSources() {
  return cancellableQuery(API, {type: 'get_dqmSources'});
}

export function getSubsystems() {
  return cancellableQuery(API, {type: 'get_subsystems'});
}

export function getSeries(dqmSource) {
  return cancellableQuery(API, {type: 'get_series', dqmSource});
}

export function getSamples(dqmSource, series) {
  return cancellableQuery(API, {type: 'get_samples', dqmSource, series});
}

export function getRuns(dqmSource, series, sample) {
  return cancellableQuery(API, {type: 'get_runs', dqmSource, series, sample});
}

export function getReferences(dqmSource, subsystem, series, sample, run) {
  return cancellableQuery(API, {type: 'get_ref', dqmSource, subsystem, series, sample, run});
}

export function loadRun(dqmSource, series, sample, run) {
  return cancellableQuery(API, {type: 'fetch_run', dqmSource, series, sample, run});
}

export function generateReport({
  dqmSource,
  subsystem,
  refSeries,
  refSample,
  refRun,
  dataSeries,
  dataSample,
  dataRun,
}, chunk_index, chunk_size) {
  return cancellableQuery(API, {
    type: 'process',
    chunk_index: chunk_index,
    chunk_size: chunk_size,
    dqmSource: dqmSource,
    subsystem: subsystem,
    ref_series: refSeries,
    ref_sample: refSample,
    ref_run: refRun,
    data_series: dataSeries,
    data_sample: dataSample,
    data_run: dataRun,
  });
}

export function queryUrl({
  dqmSource,
  subsystem,
  refSeries,
  refSample,
  refRun,
  dataSeries,
  dataSample,
  dataRun,
}) {
  const params = [
    dqmSource,
    subsystem,
    refSeries,
    refSample,
    refRun,
    dataSeries,
    dataSample,
    dataRun,
  ];
  if(!params.every(p => p)) return null;
  return `/plots/${params.join('/')}`;
}

const cancellableQuery = (endpoint, query) => {
  const source = axios.CancelToken.source();
  const p = axios
    .get(endpoint, {params: query, cancelToken: source.token})
    .then(res => {
      if (res.data.error) throw res.data.error;
      return res.data.data;
    })
    .catch(err => {
      if (axios.isCancel(err)) err.type = 'cancel';
      else console.log(err);
      throw err;
    });
  p.cancel = () => source.cancel(`Cancelled request of type ${query.type}`);
  return p;
};
