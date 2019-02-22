FROM rackspacedot/python37
COPY . /app
WORKDIR /app
RUN pip install --upgrade pip
RUN pip install pickle-mixin
RUN pip install -r requirements.txt
EXPOSE 3000
CMD python ./depressionrest.py
