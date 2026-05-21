import gdown

# Download a file
url = "https://drive.google.com/file/d/1BHCp1Ort-OaVFwaZmvsG4qHiKiPeNb4h/view?usp=sharing"

gdown.download(url=url, output="code/base_ep4k.model", quiet=False)