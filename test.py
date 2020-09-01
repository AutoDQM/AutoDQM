import ROOT
e.code =5
e.msg ='hi'
reason = e.reason if hasattr(e, 'reason') else '%d %s' % (e.code, e.msg)
