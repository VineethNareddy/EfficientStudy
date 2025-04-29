import gridfs

from pymongo import MongoClient

def mongo_conn():
    try:
        conn = MongoClient("127.0.0.1", 27017)
        db = conn['documents']
        return db
    except Exception as e:
        print("error in mongo connection: ", e)

db = mongo_conn()

def put_doc(file, db):

    file.seek(0)
    fs = gridfs.GridFS(db)
    fs.put(file, filename = file.name)

# Probably some help from ChatGPT on using libraries such as gridfs

