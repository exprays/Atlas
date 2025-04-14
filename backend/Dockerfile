FROM continuumio/miniconda3

WORKDIR /app

# Copy requirements first so they're available during env creation
COPY requirements.txt .
COPY environment.yml .

# Create conda environment
RUN conda env create -f environment.yml
ENV PATH /opt/conda/envs/myenv/bin:$PATH
RUN echo "source activate myenv" > ~/.bashrc

# Copy the rest of the code
COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
