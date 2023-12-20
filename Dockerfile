# Use Python 3.10
FROM python:3.10.9

# Set a directory for the app
WORKDIR /usr/src/app

# Copy all files to the container
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Command to run the application
CMD ["python", "./Code/app.py"]
