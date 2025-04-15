from pymongo import MongoClient
import gridfs

def mongo_conn():
    try:
        conn = MongoClient("127.0.0.1", 27017)
        db = conn['documents']
        print("mongodb connected: ", conn)
        return db
    except Exception as e:
        print("error in mongo connection: ", e)

db = mongo_conn()
file_location = "Designing Data Intensive Applications.pdf"

with open(file_location, "rb") as f:
    data = f.read()
    fs = gridfs.GridFS(db)
    fs.put(data, filename = "Designing Data Intensive Applications.pdf")
    print("upload complete")

for file in db.fs.files.find():
    print(file)
# new_data = db.fs.files.find_one({'filename': file_location})
# my_id = data['id']
# outputdata = fs.get(my_id).read()
# download_location = "C:/Users/jonat/OneDrive/Downloads/"
# output = open(download_location, "wb")
# output.write(outputdata)
# output.close()
# print("download complete")
