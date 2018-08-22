import keras
from keras.layers import Input
from keras.models import Model

def modelOut(model, layers, index):
    """Produces the outputs of the model on the device

    # Arguments
        model: keras model
        layers: list of strings representing the names of the layer in the model
        index: location of the layer where the model is split

    # Returns
        Ouputs of the device model, inputs of the remote model, strings representing the 
        names of the layers to be skipped
    """
    device = set(layers[:index+1])
    remote = layers[index+1:]

    deviceOuts = []
    remoteIns  = []
    skipNames  = []

    for i in remote:
        rIndex = layers.index(i)
        curIn = model.layers[rIndex].input
        for j in device:
            dIndex = layers.index(j)
            out = model.layers[dIndex].output
            if curIn==out:
                d = model.layers[index].output
                r = Input(out.shape[1:])
                deviceOuts.append(out)
                remoteIns.append(r)
                skipNames.append(model.layers[dIndex].name)

    return deviceOuts, remoteIns, skipNames

def createInpCfg(inp):
    cfg = {}
    cfg['name'] = inp.name.split(':')[0]
    cfg['class_name'] = 'InputLayer'
    cfg['config'] = {'batch_input_shape':tuple(inp.shape.as_list()),
    'dtype':'float32', 'sparse':False, 'name':inp.name.split(':')[0]}
    cfg['inbound_nodes'] = []

    return cfg

def createRMCfg(model, remoteIns, deviceOuts, index):
    """Create the remote model's configuration dectionary

    # Arguments
        model: keras model
        remoteIns: input tensors to the remote model
        deviceOuts: output tensors from the device model
        index: location of the layer where the model is split

    # Returns
        Dictionary representing the configuration of the remote model
    """
    deviceOuts = [i.name for i in deviceOuts]
#     remoteIns  = [i.name for i in remoteIns]
    modelCfg = model.get_config()

    remoteIns = [createInpCfg(i) for i in remoteIns]

    modelLayers = modelCfg['layers'][index+1:]

    for i in remoteIns:
        modelLayers.insert(0, i)
    modelCfg['layers'] = modelLayers

    return modelCfg

def remoteModel(model, split, custom_objects=None):
    """Creates a remote model that simulates the one run on the cloud

    # Argument
        model: keras model
        split: string represeting the name of the layer where the model is split
        custom_objects: user defined layers used in the keras model

    # Returns
        Model representing the one run in the cloud
    """
    modelLayers = [i.name for i in model.layers]
    index = modelLayers.index(split)

    deviceOuts, remoteIns, skipNames = modelOut(model, modelLayers, index)

    inNames = [i.name.split(':')[0] for i in remoteIns]

    RMCfg = createRMCfg(keras.models.clone_model(model), remoteIns, deviceOuts, index)

    for i in RMCfg['layers']:
        if len(i['inbound_nodes'])==0:
            continue
        temp = i['inbound_nodes'][0]
        for j in temp:
            if j[0] in skipNames:
                jIndex = temp.index(j)
                j[0] = inNames[skipNames.index(j[0])]
                temp[jIndex] = j
        i['inbound_nodes'][0] = temp

    RMCfg['input_layers'] = []
    for i in inNames:
        RMCfg['input_layers'].append([i, 0, 0])

    remoteModel = Model.from_config(RMCfg, custom_objects=custom_objects)

    modelLayers = [i.name for i in model.layers]

    for l in remoteModel.layers:
        orig = l.name
        if orig in modelLayers:
            lWeights = model.get_layer(orig)
            l.set_weights(lWeights.get_weights())
    return remoteModel
