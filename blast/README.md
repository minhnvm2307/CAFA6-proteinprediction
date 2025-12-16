# BLAST Search


### In /blast folder
```
# Get blastp library
wget https://ftp.ncbi.nlm.nih.gov/blast/executables/LATEST/ncbi-blast-2.17.0+-x64-linux.tar.gz

# Unzip
tar zxvpf ncbi-blast-2.17.0+-x64-linux.tar.gz

# Add to /venv/bin/
# os.environ["PATH"] += ":/ncbi-blast-2.17.0+/bin"
cp ./ncbi-blast-2.17.0+/bin/* ./venv/bin/

# Check libs
blastp -version

# Install dependencies for blast
pip install git+https://github.com/SamusRam/ProFun.git

# Run blast
python3 run_blastp.py
```

