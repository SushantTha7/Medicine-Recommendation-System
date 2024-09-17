
# Medicine Recommend System

Most people tend to live a long and healthy life, but people are busy in their
day-to-day life and it is not possible for everyone to visit doctors for minor
symptoms of a disease. Many people do not know about medicines and to visit a 
doctor and consult for minor symptoms for medicines is a time-consuming
process. AI and machine learning like emerging technology can help us to 
create a recommended system that will prescribe medicine and this system can
accurately predict a medicine to use. In this paper proposes the medicine 
recommendation system which will predict disease and medicine according to
symptoms entered by patients/users.

# Medicine Recommendation System

## Prerequisites

- Python version 3.8
- MySQL

## Step-by-Step Guide

### 1. Create a Virtual Environment

First, create a virtual environment using Python 3.8:

```bash
python3.8 -m venv <name_of_virtual_environment>
```

### 2. Activate the Virtual Environment

Activate the virtual environment:

For Windows:
```bash
<name_of_virtual_environment>/Scripts/activate
```

For Linux/macOS:
```bash
source <name_of_virtual_environment>/bin/activate
```

### 3. Install Dependencies

You can either install the required packages from the `requirements.txt` file or copy an existing `lib` folder from another virtual environment:

To install the dependencies from `requirements.txt`, run:

```bash
pip install -r requirements.txt
```

Alternatively, copy the `lib` folder from an existing virtual environment(Ghom) to your new environment.

### 4. Set Up MySQL

1. Start MySQL and create a new database:

```sql
CREATE DATABASE IF NOT EXISTS medicine;
```

2. Use the newly created database:

```sql
USE medicine;
```

3. To check the tables in the database:

```sql
SHOW TABLES;
```

### 5. Update Application Configuration

In the `application.py` file, update the database credentials (username, password, host, database name, etc.) to match your MySQL configuration.

### 6. Run the Application

Once the database is configured and the credentials are updated, run the `application.py` file to start the application:

```bash
python application.py
```

Now, your Medicine Recommendation System should be up and running!

