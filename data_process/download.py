"""
Downloads the following:
- SNLI Corpus
- MultiNLI Corpus
- SciTail Corpus
- GloVe vectors
- WordNet 
- CoreNLP tools
"""

import os
import zipfile
import tarfile

def download(url, dirpath):
    filename = url.split('/')[-1]
    filepath = os.path.join(dirpath, filename)
    os.system('wget {} -O {}'.format(url, filepath))
    return filepath

def unzip(filepath):
    print("Extracting: " + filepath)
    dirpath = os.path.dirname(filepath)
    with zipfile.ZipFile(filepath) as zf:
        zf.extractall(dirpath)
    os.remove(filepath)

def ungzip(filepath):
    print("Extracting: " + filepath)
    dirpath = os.path.dirname(filepath)
    file_zip = zipfile.ZipFile(filepath, 'r')
    for file in file_zip.namelist():
        file_zip.extract(file, dirpath)
    file_zip.close()

def untar(filepath):
    print("Extracting: " + filepath)
    dirpath = os.path.dirname(filepath)
    tar = tarfile.open(filepath)
    for file in tar.getnames():
        tar.extract(file, dirpath)
    tar.close()

def download_wordvecs(dirpath):
    if os.path.exists(dirpath):
        print('Found Glove vectors - skip')
        return
    else:
        os.makedirs(dirpath)
    url = 'http://nlp.stanford.edu/data/glove.840B.300d.zip'
    unzip(download(url, dirpath))

def download_snli(dirpath):
    if os.path.exists(dirpath):
        print('Found SNLI dataset - skip')
        return
    else:
        os.makedirs(dirpath)
    url = 'https://nlp.stanford.edu/projects/snli/snli_1.0.zip'
    unzip(download(url, dirpath))

def download_multinli(dirpath):
    if os.path.exists(dirpath):
        print('Found MultiNLI dataset - skip')
        return
    else:
        os.makedirs(dirpath)
    url = 'www.nyu.edu/projects/bowman/multinli/multinli_1.0.zip'
    ungzip(download(url, dirpath))

def download_scitail(dirpath):
    if os.path.exists(dirpath):
        print('Found SciTail dataset - skip')
        return
    else:
        os.makedirs(dirpath)
    url = 'http://data.allenai.org.s3.amazonaws.com/downloads/SciTailV1.1.zip'
    ungzip(download(url, dirpath))

def download_wordnet(dirpath):
    if os.path.exists(dirpath):
        print('Found WordNet 3.0 - skip')
        return
    else:
        os.makedirs(dirpath)
    url = 'http://wordnetcode.princeton.edu/3.0/WNprolog-3.0.tar.gz'
    untar(download(url, dirpath))

def download_corenlp(dirpath):
    if os.path.exists(dirpath):
        print('Found Stanford CoreNLP - skip')
        return
    else:
        os.makedirs(dirpath)
    url = 'http://nlp.stanford.edu/software/stanford-corenlp-full-2016-10-31.zip'
    unzip(download(url, dirpath))

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.realpath(__file__))
    snli_dir = os.path.join(base_dir, 'snli')
    multinli_dir = os.path.join(base_dir, 'multinli')
    scitail_dir = os.path.join(base_dir, 'scitail')
    wordvec_dir = os.path.join(base_dir, 'glove')
    wordnet_dir = os.path.join(base_dir, 'wordnet')
    corenlp_dir = os.path.join(base_dir, 'corenlp')
    download_snli(snli_dir)
    download_multinli(multinli_dir)
    download_scitail(scitail_dir)
    download_wordvecs(wordvec_dir)
    download_wordnet(wordnet_dir)
    download_corenlp(corenlp_dir)

