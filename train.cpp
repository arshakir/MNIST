#include <eigen3/Eigen/Dense>
#include <fstream>
#include <iostream>
#include <math.h>
#include <vector>

using Eigen::Matrix3d;
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

MatrixXd rmatrix(MatrixXd m, int v) {

  double range = 1 / sqrt(v) * 2;
  m = MatrixXd::Random(
      m.rows(),
      m.cols()); // 3x3 Matrix filled with random numbers between (-1,1)
  m = (m + MatrixXd::Constant(m.rows(), m.cols(), 1.)) * range /
      2.; // add 1 to the matrix to have values between 0 and 2; multiply with
          // range/2
  m = (m +
       MatrixXd::Constant(m.rows(), m.cols(),
                          -1 / sqrt(v))); // set LO as the lower bound (offset)

  return m;
}

void readfile(vector<MatrixXd> &images, vector<int> &targets) {
  ifstream file("mnist_train.csv");
  string snum;
  int num;
  images.resize(59900);

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

MatrixXd sigmoid(MatrixXd m) {
  for (int r = 0; r < m.rows(); r++) {
    for (int c = 0; c < m.cols(); c++) {
      m(r, c) = 1.0 / (1.0 + exp(-1 * m(r, c)));
    }
  }
  return m;
}

MatrixXd multiply(MatrixXd a, MatrixXd b) {
  MatrixXd m(a.rows(), 1);
  if (a.rows() != b.rows()) {
    cout << "ERROR";
  }

  for (int i = 0; i < a.rows(); i++) {
    m(i, 0) = a(i, 0) * b(i, 0);
  }

  return m;
}

MatrixXd sigmoidprime(MatrixXd sig) {
  return multiply(sig, (MatrixXd::Constant(sig.rows(), 1, 1) - sig));
}

network train(network net, MatrixXd input, int target) {
  MatrixXd hinputs = net.hweights * input;
  MatrixXd houtputs = sigmoid(hinputs);
  MatrixXd finputs = net.oweights * houtputs;
  MatrixXd foutputs = sigmoid(finputs);

  MatrixXd targets(10, 1);

  for (int i = 0; i < 10; i++) {
    if (target == i) {
      targets(i, 0) = 0.99;
    } else {
      targets(i, 0) = 0.01;
    }
  }

  MatrixXd oerrors = targets - foutputs;
  MatrixXd herrors = net.oweights.transpose() * oerrors;

  net.oweights =
      net.oweights + (net.rate * (multiply(oerrors, sigmoidprime(foutputs)) *
                                  houtputs.transpose()));
  net.hweights =
      net.hweights + (net.rate * (multiply(herrors, sigmoidprime(houtputs)) *
                                  input.transpose()));

  return net;
}

MatrixXd predict(network net, MatrixXd input) {
  MatrixXd hinputs = net.hweights * input;
  MatrixXd houtputs = sigmoid(hinputs);
  MatrixXd finputs = net.oweights * houtputs;
  MatrixXd foutputs = sigmoid(finputs);

  return foutputs;
}

void write(network net){
  ofstream h("mnist hweights.txt");
  h << net.hweights;
  h.close();

  ofstream o("mnist oweights.txt");
  o << net.oweights;
  o.close();
}

int main() {
  srand(time(0));

  network n;

  n.inputs = 784;
  n.hiddens = 200;
  n.outputs = 10;
  n.rate = 0.1;
  n.hweights.resize(n.hiddens, n.inputs);
  n.oweights.resize(n.outputs, n.hiddens);
  n.hweights = rmatrix(n.hweights, n.inputs);
  n.oweights = rmatrix(n.oweights, n.hiddens);

  vector<MatrixXd> images;
  vector<int> targets;

  readfile(images, targets);

  
  for (int i = 0; i < images.size(); i++) {
    if (i % 100 == 0) {
      cout << "img: " << i << endl;
    }
    n = train(n, images[i], targets[i]);
  }
  


  cout << predict(n, images[203]) << endl << targets[203];
  write(n);
}
