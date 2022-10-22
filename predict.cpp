#include <eigen3/Eigen/Dense>
#include <fstream>
#include <iostream>
#include <math.h>
#include <vector>

using Eigen::MatrixXd;
using namespace std;


struct network {
  int inputs;
  int hiddens;
  int outputs;
  double rate;

  MatrixXd hweights;
  MatrixXd oweights;
};


MatrixXd sigmoid(MatrixXd m) {
  for (int r = 0; r < m.rows(); r++) {
    for (int c = 0; c < m.cols(); c++) {
      m(r, c) = 1.0 / (1.0 + exp(-1 * m(r, c)));
    }
  }
  return m;
}


MatrixXd predict(network net, MatrixXd input) {
  MatrixXd hinputs = net.hweights * input;
  MatrixXd houtputs = sigmoid(hinputs);
  MatrixXd finputs = net.oweights * houtputs;
  MatrixXd foutputs = sigmoid(finputs);

  return foutputs;
}

MatrixXd readweights(string name, int rows, int cols){
  MatrixXd m(rows, cols);
  ifstream file(name);

  for (int r=0; r < rows; r++){
    for (int c=0; c < cols; c++){
      file >> m(r,c);
    }
  }

  file.close();

  return m;
}

void readfile(vector<MatrixXd> &images, vector<int> &targets) {
  ifstream file("mnist_test.csv");
  string snum;
  int num;
  images.resize(10000);

  for (int i = 0; i < images.size(); i++) {
    images[i].resize(784, 1);
    getline(file, snum, ',');
    num = stoi(snum);
    targets.push_back(num);
    for (int a = 0; a < 784; a++) {
      getline(file, snum, ',');
      num = stoi(snum);
      if (num == 0) {
        images[i](a, 0) = 0;
      } else {
        images[i](a, 0) = 1;
      }
    }
  }

  cout << "images loaded" << endl;
  file.close();
}
int main(){
  network n;

  n.inputs = 784;
  n.hiddens = 200;
  n.outputs = 10;
  n.rate = 0.1;

  n.oweights = readweights("mnist oweights.txt", 10, 200);
  n.hweights = readweights("mnist hweights.txt", 200, 784);

  vector<MatrixXd> images;
  vector<int> targets;

  readfile(images, targets);

  MatrixXd output;
  int num, fake;
  int correct=0;

  for (int i=0; i < images.size(); i++){
    output = predict(n, images[i]);
    output.maxCoeff(&num, &fake);
    if (num == targets[i]){
      correct++;
    }
  }

  cout << correct << "/" << images.size() << endl;
}


