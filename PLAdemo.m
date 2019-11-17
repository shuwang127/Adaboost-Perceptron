clear;
close all;

%% read data.
[ data, label ] = readdata();

%% parameters.
iter = 20;
ncls = 30;

%% PLA
[ w, Ein, acc ] = PLAtrain( data, label, iter );
figure(1);
plot(Ein); xlabel('iterative'); ylabel('in-sample error');
disp(['The training accuracy for perceptron : ', num2str(acc)]);

%% Adaboost for PLA
[ weights, alphas, accplot ] = PLAboosttrain( data, label, ncls, iter );
figure(2);
plot(accplot); xlabel('number of classifiers'); ylabel('accuracy');
disp(['The training accuracy for perceptron (Adaboost) : ', num2str(accplot(end))]);