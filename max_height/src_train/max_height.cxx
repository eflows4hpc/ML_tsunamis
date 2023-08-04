#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <mpi.h>
#include "eddl.cxx"
#include "eddl/serialization/onnx/eddl_onnx.h"

void mostrarFormatoPrograma(char *argv[])
{
	cerr << "Use:" << endl;
	cerr << argv[0] << " dataFile" << endl << endl; 
	cerr << "dataFile format:" << endl;
	cerr << "  File name of the network inputs" << endl;
	cerr << "  File name of the network outputs" << endl;
	cerr << "  Initializacion of the train, validation and test sets (" << EXACT << ": exact, " << PERCENTAGE << ": percentage)" << endl;
	cerr << "  If " << EXACT << ":" << endl;
	cerr << "    A line specifying the number of samples in the train, validation and test sets" << endl;
	cerr << "  Else if " << PERCENTAGE << ":" << endl;
	cerr << "    A line specifying the percentage of samples in the train, validation and test sets" << endl;
	cerr << "  Number of hidden layers" << endl;
	cerr << "  Number of units per hidden layer" << endl;
	cerr << "  Maximum number of epochs" << endl;
	cerr << "  Batch size" << endl;
	cerr << "  Patience for the early stopping" << endl;
	cerr << "  Initial learning rate" << endl;
	cerr << "  Reduce learning rate on plateau (0: no, 1: yes)" << endl;
	cerr << "  If 1:" << endl;
	cerr << "    Patience" << endl;
	cerr << "    Final learning rate" << endl;
	cerr << "  Output ONNX file prefix" << endl;
}

int main(int argc, char **argv)
{
	int i, j;
	int err = 0;
	int err_global;
	double val, val_min, factor;
	bool use_cpu = true;
	string directorio, fich_ent;
	string fich_datosX, fich_datosY;
	string fich_onnx;
	int flag_sets;
	int num_ejemplos;
	int num_train, num_val, num_test;
	float porc_train, porc_val, porc_test;
	int num_puntos;
	int num_capas, num_unidades;
	int max_epochs, epochs;
	int batch_size;
    int patience_es, patience_lr;
	int reducir_lr;
	float lr_inicial, lr_final;
	int id_hebra, num_procs;
	int id_mejor;
	int seed;
	char cadena_random[16];
	char cadena_loss[16];

	Tensor *x_train, *y_train;
	Tensor *x_val, *y_val;
	Tensor *x_test, *y_test;
	float mejor_loss;
	model net;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &id_hebra);
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

	if (argc < 2) {
		if (id_hebra == 0)
			mostrarFormatoPrograma(argv);
		MPI_Finalize();
		return EXIT_SUCCESS;
	}

	// Fichero de datos
	fich_ent = argv[1];
	if (! existeFichero(fich_ent)) {
		if (id_hebra == 0)
			cerr << "Error: File '" << fich_ent << "' not found" << endl;
		err = 1;
	}

	if (err == 0) {
		// Leer los datos del problema
		err = cargarDatosProblema(fich_ent, fich_datosX, fich_datosY, directorio, &flag_sets, &num_train, &num_val, &num_test,
				&porc_train, &porc_val, &porc_test, &num_capas, &num_unidades, &max_epochs, &batch_size, &patience_es,
				&lr_inicial, &reducir_lr, &patience_lr, &lr_final, fich_onnx, id_hebra);
	}
	MPI_Allreduce(&err, &err_global, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
	if (err_global != 0) {
		MPI_Finalize();
		return EXIT_FAILURE;
	}

	// Cargar todos los datos en x_train, y_train, x_val, y_val, x_test e y_test
	num_puntos = obtener_num_puntos(directorio, fich_datosY);
	cargar_datos(directorio, fich_datosX, fich_datosY, flag_sets, &num_ejemplos, num_train, num_val, num_test, porc_train,
		porc_val, porc_test, num_puntos, 1234, &x_train, &y_train, &x_val, &y_val, &x_test, &y_test, &val_min, &factor);

	if (id_hebra == 0) {
		mostrarDatosProblema(flag_sets, num_ejemplos, num_train, num_val, num_test, porc_train, porc_val, porc_test,
			num_puntos, num_capas, num_unidades, max_epochs, batch_size, patience_es, lr_inicial, reducir_lr,
			patience_lr, lr_final);
	}

    // Definir la red neuronal
	float *losses = new float[num_procs];
	struct timeval tv;
	gettimeofday(&tv, NULL);
	seed = tv.tv_usec;
	build_model(&net, lr_inicial, use_cpu, seed, num_puntos, num_capas, num_unidades, id_hebra);

    // Entrenamiento con early stopping
	mejor_loss = entrenamiento(net, x_train, y_train, x_val, y_val, num_puntos, max_epochs, batch_size,
					patience_es, patience_lr, lr_final, &epochs, id_hebra);

	MPI_Gather(&mejor_loss, 1, MPI_FLOAT, losses, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

	// El proceso 0 pone en id_mejor el id del proceso con el mejor modelo
	// y escribe los parámetros de normalización del modelo
	if (id_hebra == 0) {
		val = losses[0];
		id_mejor = 0;
		for (i=0; i<num_procs; i++) {
			if (losses[i] < val) {
				val = losses[i];
				id_mejor = i;
			}
		    fprintf(stdout, "Process %d, loss = %e\n", i, losses[i]);
		}
		escribirParametrosEntrenamiento((char *) "inference_parameters.txt", val_min, factor);
	}

	MPI_Bcast(&id_mejor, 1, MPI_INT, 0, MPI_COMM_WORLD);

	if (id_hebra == id_mejor) {
	    fprintf(stdout, "\nResults best model\n");
		// El proceso con el mejor modelo escribe el resultado por pantalla y guarda el modelo
		evaluar_modelo(net, x_train, y_train, x_val, y_val, x_test, y_test, num_puntos, factor);
		generarCadenaAleatoria(cadena_random, 8);
		sprintf(cadena_loss, "%e", mejor_loss);
		fich_onnx = fich_onnx + "_" + cadena_random + "_" + cadena_loss + ".onnx";
		save_net_to_onnx_file(net, fich_onnx.c_str());
		guardarMetadatosEnFicheroONNX(fich_onnx.c_str(), epochs, batch_size, patience_es, lr_inicial,
			reducir_lr, patience_lr, lr_final);
	}

	MPI_Finalize();
	delete losses;
	delete x_train;
	delete y_train;
	delete x_val;
	delete y_val;
	delete x_test;
	delete y_test;
	delete net;

	return EXIT_SUCCESS;
}

