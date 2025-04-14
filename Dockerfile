FROM cern/cc7-base
EXPOSE 8083

# Update the EPEL repository URLs
RUN sed -i 's#http://linuxsoft.cern.ch/epel/7/#http://linuxsoft.cern.ch/internal/archive/epel/7/#g' /etc/yum.repos.d/epel.repo

# Update packages and install dependencies
RUN yum update -y && yum install -y \
      ImageMagick \
      httpd \
      npm \
      php \
      python3-pip  

# Set python alias
RUN echo "alias python=python3" >>~/.bashrc

# Install additional packages
RUN yum update -y && yum install -y \
      epel-release \
      root \
      python3-root

# Install Python requirements
COPY requirements.txt /code/requirements.txt
RUN pip3 install -r /code/requirements.txt

RUN mkdir /db /run/secrets
RUN chown -R 1000:1000 /db /var/www /run/secrets
RUN chmod -R 777 /db

RUN ln -s /dev/stdout /home/access_log
RUN ln -s /dev/stderr /home/error_log

RUN chown 1000:1000 /home/error_log
RUN chown 1000:1000 /home/access_log   
RUN chmod 777  /home/error_log
RUN chmod 777 /home/access_log  

ENV HOME /root
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
# ENV ADQM_MODULES /var/www/cgi-bin/modules/

WORKDIR /webapp
COPY webapp/package.json /webapp/package.json
RUN npm install

COPY webapp /webapp
RUN npm run build
RUN cp -r /webapp/build /var/www/public
RUN cp -r /webapp/build /webapp/public

RUN mkdir /var/www/results /var/www/results/pdfs /var/www/results/pngs /var/www/results/jsons
RUN chmod 777 /var/www/results /var/www/results/pdfs /var/www/results/pngs /var/www/results/jsons

COPY httpd.conf /etc/httpd/conf/httpd.conf
COPY index.py /var/www/cgi-bin/index.py
COPY autodqm /var/www/cgi-bin/autodqm
COPY autoref /var/www/cgi-bin/autoref
COPY plugins /var/www/cgi-bin/plugins
COPY models /var/www/cgi-bin/models
# COPY modules /var/www/cgi-bin/modules
COPY config /var/www/public/config

RUN chgrp -R 1000 /run && chmod -R g=u /run
RUN chgrp -R 1000 /etc/httpd/logs && chmod -R g=u /etc/httpd/logs

CMD ["/usr/sbin/httpd","-D","FOREGROUND"]
