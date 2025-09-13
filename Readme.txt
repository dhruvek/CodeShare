nohup python3 TTS_V3_Deploy.py >> /home/ubuntu/D_Test/REPO/V1/logs/console.log 2>&1 &

pkill -f 'python3 TTS_V3_Deploy.py' || true

sudo lsof -i :8009