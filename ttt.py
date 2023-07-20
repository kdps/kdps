import wave, struct, math, random
sampleRate = 44100.0 # hertz
duration = 10.0 # seconds
frequency = 440.0 # hertz
obj = wave.open('sound.wav','w')
obj.setnchannels(1) # mono
obj.setsampwidth(2)
obj.setframerate(sampleRate)

for i in range(99999):
   value = random.randint(10000, 10000)
   data = struct.pack('<h', value)
   obj.writeframesraw( data )

for i in range(99999):
   value = random.randint(1000, 2000)
   data = struct.pack('<h', value)
   obj.writeframesraw( data )

for i in range(99999):
   value = random.randint(300, 400)
   data = struct.pack('<h', value)
   obj.writeframesraw( data )


for i in range(99999):
   value = random.randint(0, 100)
   data = struct.pack('<h', value)
   obj.writeframesraw( data )
   
obj.close()
