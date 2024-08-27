FROM python:3.9

# Set the working directory for the application
WORKDIR /home/user/app

# Copy the requirements file
COPY ./requirements.txt ./requirements.txt

# Install the dependencies
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Create a new user and switch to it
RUN useradd -m user

# Create the cache directory and set permissions
RUN mkdir -p /home/user/.cache && chown -R user:user /home/user/.cache

USER user

# Set environment variables
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
    ENV HF_HOME=/home/user/.cache

# Copy the application files and set ownership to the new user
COPY --chown=user . .

# Expose the port that your application runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
