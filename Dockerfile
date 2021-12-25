FROM python:3.9

COPY requirements.txt .
COPY . /home/Stock_Classification_project

RUN pip install -r requirements.txt

WORKDIR /home/Stock_Classification_project

CMD ["bash"]