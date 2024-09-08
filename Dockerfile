FROM public.ecr.aws/docker/library/python:3.10-slim-bullseye

RUN pip3 install joblib threadpoolctl pandas numpy scikit-learn==1.4.2
ENV PYTHONUNBUFFERED=TRUE

ENTRYPOINT ["python3"]
