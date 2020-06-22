import dropbox as db

# Dropbox Developer에서 필요한 토큰.
access_token = "U5cSTVPcbrAAAAAAAAAAFyJqqvRh24lYvq5oMP-w2YFGo0b2QiZZgaXTD7mikOx8"

# instance 생성
dbx = db.Dropbox(access_token)

# 연결 상태 확인
print(dbx.users_get_current_account())

#filename = "테스트파일.txt"
#pathname = "/테스트폴더/테스트파일.txt"
#with open(filename, "rb") as f:
#   dbx.files_upload(f.read(), pathname, mode=dropbox.files.WriteMode.overwrite)

# root directory에 있는 파일들 확인
for entry in dbx.files_list_folder('').entries:
    print(entry.name)

# 업로드
json_text = {
    'hello':'world',
    'good':123
}
print(dbx.files_upload(str(json_text).encode('utf-8'), '/test.json'))


# 다운로드
with open('exex.png', "wb") as f:
    metadata, res = dbx.files_download(path="/Users/hcy/Desktop")
    f.write(res.content)
