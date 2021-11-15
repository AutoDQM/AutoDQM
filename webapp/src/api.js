import axios from 'axios';

const API = './cgi-bin/index.py';

export function getSubsystems() {
  return cancellableQuery(API, {type: 'get_subsystems'});
}

export function getSeries() {
  return cancellableQuery(API, {type: 'get_series'});
}

export function getSamples(series) {
  return cancellableQuery(API, {type: 'get_samples', series});
}

export function getRuns(series, sample) {
  return cancellableQuery(API, {type: 'get_runs', series, sample});
}

export function getReferences(subsystem, series, sample, run) {
  return cancellableQuery(API, {type: 'get_ref', subsystem, series, sample, run});
}

export function loadRun(series, sample, run) {
  return cancellableQuery(API, {type: 'fetch_run', series, sample, run});
}

export function generateReport({
  subsystem,
  refSeries,
  refSample,
  refRun,
  dataSeries,
  dataSample,
  dataRun,
}) {
  return cancellableQuery(API, {
    type: 'process',
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
  subsystem,
  refSeries,
  refSample,
  refRun,
  dataSeries,
  dataSample,
  dataRun,
}) {
  const params = [
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
