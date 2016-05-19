#include <iostream>
#include <cstdlib>
#include <cstring>
#include <random>
#include <time.h>

#include "Header.h"

#include "Eigen\Eigen"

using namespace Eigen;

extern "C"
{	
	EXPORT int Hello42()
	{
		
		return 42;
	}

	EXPORT double randInRange(double min, double max)
	{

		return min + ((rand() / (double)RAND_MAX) * (max - min));
	}

	EXPORT void removeModel(double * model)
	{
		free(model);
	}

	EXPORT double doubleRandInRange(double min, double max)
	{		
		return min + ((rand() / (double)RAND_MAX) * (max - min));
	}

	EXPORT int intRandInRange(int min, int max) {
		return (int)floor(doubleRandInRange((double)min, max + 1));
	}

	void removeLinearModel(double * model)
	{
		free(model);
	}

	EXPORT double * createModelLinear(int nbInput)
	{
		double * model = (double *)malloc((nbInput + 1) * sizeof(double));

		model[0] = 1; // neurone de biais

		srand((unsigned)time(NULL));
		for (int i = 1; i <= nbInput; ++i)
		{
			model[i] = randInRange(-1, 1);
		}
		return model;
	}

	EXPORT int classifyLinear(double * model, double * element, int nbInput)
	{
		double sum = model[0]; // neurone de biais
		for (int i = 0; i < nbInput; ++i)
		{
			sum += model[i + 1] * element[i];
		}
		return sum < 0 ? -1 : 1;
	}

	EXPORT void trainPLA(double * exemples, int nbInput, int nbExemples, int * expectedResult, double * model, int maxIteration)
	{
		int itNb = 0;
		while (itNb < maxIteration)
		{
			bool allAreWellClassified = true;
			for (int i = 0; i < nbExemples; ++i)
			{
				int expectedRes = expectedResult[i];

				if (expectedRes != classifyLinear(model, &exemples[i * nbInput], nbInput))
				{
					allAreWellClassified = false;

					model[0] += 0.1 * expectedRes;

					for (int j = 0; j < nbInput; ++j)
					{
						model[j + 1] = model[j + 1] + 0.1 * expectedRes * exemples[i * nbInput + j];
					}
				}
			}

			if (allAreWellClassified)
			{
				return;
			}

			++itNb;
		}
	}

	EXPORT double classifyRegression(double* model, double* input, int inputSize)
	{	
		double sum = model[0]; // neurone de biais
		for (int i = 0; i < inputSize; ++i)
		{
			sum += model[i + 1] * input[i];
		}
		return sum; 
	}


	EXPORT double sign(double * x, int taillex, double * w) {
		double sx = w[0];
		for (int i = 0; i < taillex; i++) {
			sx += x[i] * w[i + 1];
		}
		return sx >= 0 ? 1 : -1;
	}

	EXPORT void perceptron_regression(double* model, double* inputs, int inputCount, int inputSize, double* output)
	{
		Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Y(inputCount, 1);

		for (int i = 0; i < inputCount; ++i)
		{
			Y(i, 0) = *output;
			++output;
		}

		Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> X(inputCount, inputSize + 1);

		const int jFinal = inputSize + 1;

		for (int i = 0; i < inputCount; ++i)
		{
			X(i, 0) = 1.0;

			for (int j = 1; j < jFinal; ++j)
			{
				X(i, j) = *inputs;
				++inputs;
			}
		}

		Eigen::MatrixXd Xt = X.transpose();

		Eigen::MatrixXd W = ((Xt * X).inverse() * Xt) * Y;

		for (int i = 0; i < W.rows(); ++i)
		{
			*model = W(i, 0);
			++model;
		}
	}
	EXPORT double *** createNetworkModel(int * nbDim, int nbLayer)
	{
		srand((unsigned)time(NULL));

		double *** model = (double ***)malloc(nbLayer * sizeof(double **));

		/*model[0] = (double **) malloc((nbDim[0] + 1) * sizeof(double *));
		for(int neur = 0; neur < (nbDim[0] + 1); ++neur) {
		model[0][neur] = (double *) malloc(sizeof(double));
		model[0][neur][0] = 1; //neurone de biais
		}*/

		for (int layer = 1; layer < nbLayer; ++layer) {
			model[layer] = (double **)malloc((nbDim[layer] + 1) * sizeof(double *));
			for (int neur = 0; neur < (nbDim[layer] + 1); ++neur) {
				model[layer][neur] = (double *)malloc((nbDim[layer - 1] + 1) * sizeof(double));
				for (int weight = 0; weight < nbDim[layer - 1] + 1; ++weight) {
					model[layer][neur][weight] = doubleRandInRange(-1, 1);
				}
			}
		}

		return model;
	}

	EXPORT void removeNetworkModel(double *** model, int * nbDim, int nbLayer)
	{
		for (int layer = 1; layer < nbLayer; ++layer)
		{
			for (int neuron = 0; neuron < nbDim[layer]; ++neuron)
			{
				free(model[layer][neuron]);
			}
			free(model[layer]);
		}
		free(model);
	}

	EXPORT double ** create2DModel(int * nbDim, int nbLayer)
	{
		double ** model = (double **)malloc(nbLayer * sizeof(double *));

		for (int i = 0; i < nbLayer; ++i)
		{
			model[i] = (double *)calloc((size_t)nbDim[i] + 1, sizeof(double));
		}

		return model;
	}

	EXPORT void remove2DModel(double ** model, int nbLayer)
	{
		for (int i = 0; i < nbLayer; ++i)
		{
			free(model[i]);
		}
		free(model);
	}

	EXPORT double matrixProduct(double * weights, double * outputs, int nbLink)
	{
		double p = 0;
		for (int i = 0; i < nbLink; ++i)
		{
			p += weights[i] * outputs[i];
		}
		return p;
	}

	EXPORT void computeOutput(double ** outputs, double *** network, int * neuronsCount, double * sample, int nbLayer)
	{
		for (int j = 0; j < nbLayer; ++j)
		{
			outputs[j][0] = 1;
		}

		for (int j = 0; j < neuronsCount[0]; ++j)
		{
			outputs[0][j + 1] = sample[j];
		}

		for (int layer = 1; layer < nbLayer; ++layer) {
			for (int neuron = 1; neuron <= neuronsCount[layer]; ++neuron) {
				outputs[layer][neuron] = tanh(
					matrixProduct(network[layer][neuron], outputs[layer - 1], neuronsCount[layer - 1] + 1)
				);
			}
		}
	}

	EXPORT void computeDelta(double **  gradients, double **  outputs, int expectedResult, double *** network, int * neuronsCount, int nbLayer)
	{
		for (int layer = nbLayer - 1; layer > 0; --layer) {
			for (int neuron = 1; neuron <= neuronsCount[layer]; ++neuron) {

				if (layer == nbLayer - 1)
				{

					double output = outputs[layer][neuron];
					gradients[layer][neuron] = (1 - pow(output, 2)) * (output - expectedResult);

				}
				else
				{

					double propagatedSigma = 0;
					for (int nr = 1; nr <= neuronsCount[layer + 1]; ++nr) {
						propagatedSigma += network[layer + 1][nr][neuron] * gradients[layer + 1][nr];
					}

					propagatedSigma *= 1 - pow(outputs[layer][neuron], 2);

					gradients[layer][neuron] = propagatedSigma;

				}
			}
		}
	}

	EXPORT void computeWeight(double ** gradients, double ** outputs, double alpha, double *** network, int * neuronsCount, int nbLayer)
	{
		for (int layer = 1; layer < nbLayer; ++layer)
		{
			for (int neuron = 1; neuron <= neuronsCount[layer]; ++neuron)
			{
				double gradient = gradients[layer][neuron];

				for (int weight = 0; weight <= neuronsCount[layer - 1]; ++weight)
				{
					double output = outputs[layer - 1][weight];
					network[layer][neuron][weight] -= alpha * output * gradient;
				}
			}
		}
	}

	EXPORT void trainMLP(double * exemples, int nbExemples, int * expectedResult, double *** network, int * neuronsCount, int nbLayer, int maxIteration, double alpha)
	{
		int itNb = 0;

		double **outputs = create2DModel(neuronsCount, nbLayer);
		double **gradients = create2DModel(neuronsCount, nbLayer);

		while (itNb < maxIteration) {

			int i = intRandInRange(0, nbExemples - 1);
			double * currentExemple = &exemples[i * neuronsCount[0]];

			computeOutput(outputs, network, neuronsCount, currentExemple, nbLayer);

			computeDelta(gradients, outputs, expectedResult[i], network, neuronsCount, nbLayer);

			computeWeight(gradients, outputs, alpha, network, neuronsCount, nbLayer);

			++itNb;
		}

		remove2DModel(outputs, nbLayer);
		remove2DModel(gradients, nbLayer);
	}

	EXPORT int * classifyMLP(double *** network, double * element, int * neuronsCount, int nbLayer)
	{
		double ** outputs = create2DModel(neuronsCount, nbLayer);

		computeOutput(outputs, network, neuronsCount, element, nbLayer);

		int nbFinalOutputs = neuronsCount[nbLayer - 1];
		int * finalOutputs = (int *)malloc(sizeof(int) * nbFinalOutputs);
		for (int i = 0; i < nbFinalOutputs; ++i) {
			finalOutputs[i] = outputs[nbLayer - 1][i + 1] > 0 ? 1 : -1;
		}

		remove2DModel(outputs, nbLayer);

		return finalOutputs;
	}
}