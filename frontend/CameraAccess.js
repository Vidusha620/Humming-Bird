import React, { useState, useEffect } from 'react';
import { StatusBar } from 'expo-status-bar';
import { StyleSheet, Text, View, TouchableOpacity, Image } from 'react-native';
import { useFonts } from 'expo-font';
import { Camera } from 'expo-camera';
  
export default function App() {
  const [fontsLoaded] = useFonts({
    'Inter-Topic': require('./assets/Fonts/Inter/static/Inter-Bold.ttf'),
    'Inter-Body': require('./assets/Fonts/Inter/static/Inter-Regular.ttf')
  });

  const [isChecked, setIsChecked] = useState(false);
  const [hasPermission, setHasPermission] = useState(null);
  const [cameraOpen, setCameraOpen] = useState(false);

  useEffect(() => {
    (async () => {
      const { status } = await Camera.requestCameraPermissionsAsync();
      setHasPermission(status === 'granted');
    })();
  }, []);

  const AllowPress = async () => {
    if (hasPermission) {
      setCameraOpen(true);
    } else {
      // Handle no camera permission granted
      console.log("Camera permission not granted!");
    }
  };

  if (!fontsLoaded) {
    return null;
  }

  if (cameraOpen) {
    return (
      <View style={{ flex: 1 }}>
        <Camera style={{ flex: 1 }} type={Camera.Constants.Type.back} />
        <StatusBar style="auto" />
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <Image style={styles.cameraImage} source={require('./assets/Images/camera 1.png')} />
      <Text style={styles.topic}>ACCESS REQUIRED</Text>
      <Text style={styles.paragraph}>The Camera Access feature in the Humming Bird Sign Language Translator App allows users to interact with sign language in real-time. Seamlessly integrated into the app's interface, the Camera feature enables users to capture and analyze sign language gestures on the spot. Users can point their device's camera towards a signer, and the app will provide instant feedback, aiding in the understanding and learning of sign language. The intuitive design ensures effortless access to the Camera feature, empowering users to engage dynamically with sign language expressions. Whether practicing or learning on the go, the Camera Access feature in the Humming Bird app offers a dynamic and interactive way to enhance sign language skills.</Text>
      <TouchableOpacity style={styles.button} onPress={AllowPress}>
        <Text style={styles.buttonText}>ALLOW</Text>
      </TouchableOpacity>
      <StatusBar style="auto" />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#3A0BA3',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 20,
  },
  topic: {
    fontFamily: 'Inter-Topic',
    color: '#fff',
    fontSize: 20,
    marginBottom: 30,
    textAlign: 'center',
  },
  paragraph: {
    fontFamily: 'Inter-Body',
    color: '#fff',
    textAlign: 'justify',
    marginBottom: 60,
    padding: 20
  },
  button: {
    backgroundColor: '#fff',
    paddingVertical: 10,
    paddingHorizontal: 30,
    borderRadius: 50,
    marginBottom: 180,
  },
  buttonText: {
    fontFamily: 'Inter-Topic',
    color: '#3A0BA3',
    fontSize: 20,
  },
  cameraImage: {
    resizeMode: 'contain',
    width: 70,
    bottom: -90
  }
});
