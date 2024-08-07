package org.example;

import javax.swing.*;
import java.awt.*;

public class Main extends javax.swing.JFrame{
    public static void main(String[] args) throws Exception{
//        JTextField myOutput = new JTextField(16);
//        myOutput.setText("some text");
        JFrame frame = new JFrame("Demo program for JFrame");
        frame.setLayout(new GridBagLayout());
        frame.setSize(300, 200);
        frame.setVisible(true);
        JLabel label = new JLabel("JFrame By Example");
        frame.add(label);
        Thread.sleep(3000);
        JLabel label1 = new JLabel("Nextdflnmf"+"gfgdffb");
        frame.add(label1);
        frame.invalidate();
        frame.validate();
        frame.repaint();
        Thread.sleep(3000);
        frame.remove(label);
        frame.invalidate();
        frame.validate();
        frame.repaint();
        Thread.sleep(3000);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        System.exit(0);

    }
}