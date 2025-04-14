FROM continuumio/miniconda3

WORKDIR /app

# Create conda env and install dependencies
COPY environment.yml .
RUN conda env create -f environment.yml
RUN echo "source activate myenv" > ~/.bashrc
ENV PATH /opt/conda/envs/myenv/bin:$PATH

# Copy app
COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
