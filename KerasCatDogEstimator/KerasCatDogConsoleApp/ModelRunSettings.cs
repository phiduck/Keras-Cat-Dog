using System;
using System.Collections.Generic;
using System.Text;

namespace KerasCatDogConsoleApp
{

    /// <summary>
    /// Stores the properties we will use to create a pipeline to load in the keras model.
    /// 
    /// The following command can be ran in the directory of your saved model to determine the
    /// image width and height as well as the names of the input and output layers 
    /// 
    /// > saved_model_cli show --dir KerasModel --all
    /// </summary>
    public class ModelRunSettings
    {
        /// <summary>
        /// Directory containing the model .pb file and 
        /// </summary>
        public string ModelLocation { get; set; }

        /// <summary>
        /// Directory containing the location of the data we wish to pass through the model
        /// </summary>
        public string DataLocation { get; set; }

        /// <summary>
        /// The height of the image passed into the input layer of the model
        /// </summary>
        public int ImageHeight { get; set; }

        /// <summary>
        /// The width of the image passed into the input layer of the model
        /// </summary>
        public int ImageWidth { get; set; }

        /// <summary>
        /// The name of the input layer of the keras model
        /// </summary>
        public string KerasModelInputColumnName { get; set; }
        
        /// <summary>
        /// The name of the output layer of the keras model
        /// </summary>
        public string KerasModelOutputColumnName { get; set; }
    }
}
