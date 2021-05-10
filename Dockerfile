FROM cern/cc7-base
EXPOSE 8080

RUN yum update -y && yum install -y \
      ImageMagick \
      httpd \
      npm \
      php \
      python3-pip  


RUN echo "alias python=python3" >>~/.bashrc

RUN yum update -y && yum install -y \
      epel-release \
      root \
      python3-root


COPY requirements.txt /code/requirements.txt
RUN pip3 install -r /code/requirements.txt

RUN mkdir /db /run/secrets
RUN chown -R apache:apache /db /var/www /run/secrets

RUN ln -s /dev/stdout /etc/httpd/logs/access_log
RUN ln -s /dev/stderr /etc/httpd/logs/error_log

ENV REQUESTS_CA_BUNDLE /etc/ssl/certs/ca-bundle.crt
ENV ADQM_SSLCERT /run/secrets/cmsvo-cert.pem
ENV ADQM_SSLKEY /run/secrets/cmsvo-cert.key
RUN mkdir -p /var/adqm
ENV ADQM_TMP /var/adqm
ENV ADQM_DB /db/
ENV ADQM_PUBLIC /var/www/
ENV ADQM_CONFIG /var/www/public/config/
ENV ADQM_PLUGINS /var/www/cgi-bin/plugins/
ENV ADQM_MODELS /var/www/cgi-bin/models/
ENV ADQM_MODULES /var/www/cgi-bin/modules/

WORKDIR /webapp
COPY webapp/package.json /webapp/package.json
RUN npm install

COPY webapp /webapp
RUN npm run build
RUN cp -r /webapp/build /var/www/public

COPY httpd.conf /etc/httpd/conf/httpd.conf
COPY index.py /var/www/cgi-bin/index.py
COPY autodqm /var/www/cgi-bin/autodqm
COPY autoref /var/www/cgi-bin/autoref
COPY plugins /var/www/cgi-bin/plugins
COPY models /var/www/cgi-bin/models
COPY modules /var/www/cgi-bin/modules
COPY config /var/www/public/config

CMD ["/usr/sbin/httpd","-D","FOREGROUND"]

