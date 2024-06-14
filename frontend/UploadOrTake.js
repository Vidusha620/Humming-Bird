import React, { useState } from 'react';
import { StatusBar } from 'expo-status-bar';
import { StyleSheet, Text, View, TouchableOpacity, Image } from 'react-native';
import { useFonts } from 'expo-font';

import { useNavigation } from '@react-navigation/native';

export default function App() {
  const [fontsLoaded] = useFonts({
    'Inter-Topic': require('./assets/Fonts/Inter/static/Inter-Bold.ttf'),
    'Inter-Body': require('./assets/Fonts/Inter/static/Inter-Regular.ttf')
  });

  const [isChecked, setIsChecked] = useState(false);

  const navigation = useNavigation();

  if (!fontsLoaded) {
    return null;
  }

  const CameraPress = ()=> {
    navigation.navigate('CameraAccess')
  }

  const GalleryPress = ()=> {
    navigation.navigate('GalleryAccess')
  }

  return (
    <View style={styles.container}>
      <TouchableOpacity style={styles.button1} onPress={(CameraPress)}>
        <Image style={styles.buttonImage} source={require('./assets/Images/camera 1.png')} />
      </TouchableOpacity>
      <Text style={styles.topic}>UPLOAD / TAKE</Text>
      <TouchableOpacity style={styles.button2} onPress={(GalleryPress)}>
        <Image style={styles.buttonImage} source={require('./assets/Images/upload 1.png')} />
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
    fontSize: 25,
    marginBottom: 50,
    textAlign: 'center',
  },
  buttonImage: {
    resizeMode: 'contain',
    width: 60,
    
  }
});