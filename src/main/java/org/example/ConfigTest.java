package org.example;

import java.io.FileInputStream;
import java.util.Properties;

public class ConfigTest {

    public static void main(String[] Args) throws Exception{
        Properties prop = new Properties();
        String fileName = "parameters.config";
        FileInputStream fis = new FileInputStream(fileName);
        prop.load(fis);
        int channels = Integer.valueOf(prop.getProperty("channels"));
        System.out.println(channels);

    }
}
