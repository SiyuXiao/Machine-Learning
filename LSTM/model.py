import numpy
import function


class Model(object):
    
    def forward(self):
        raise NotImplementedError

    def error(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError


class LSTM(Model):

    def __init__(self, inputsize, hiddensize, outputsize):
        self.parameter = dict(wfh=numpy.random.uniform(low=-numpy.sqrt(6. / (inputsize + hiddensize + hiddensize)),
                                                       high=numpy.sqrt(6. / (inputsize + hiddensize + hiddensize)),
                                                       size=(hiddensize, hiddensize)),
                              wfx=numpy.random.uniform(low=-numpy.sqrt(6. / (inputsize + hiddensize + hiddensize)),
                                                       high=numpy.sqrt(6. / (inputsize + hiddensize + hiddensize)),
                                                       size=(hiddensize, inputsize)),
                              bf=numpy.random.uniform(low=-numpy.sqrt(6. / (inputsize + hiddensize + hiddensize)),
                                                      high=numpy.sqrt(6. / (inputsize + hiddensize + hiddensize)),
                                                      size=(hiddensize, )),
                              wih=numpy.random.uniform(low=-numpy.sqrt(6. / (inputsize + hiddensize + hiddensize)),
                                                       high=numpy.sqrt(6. / (inputsize + hiddensize + hiddensize)),
                                                       size=(hiddensize, hiddensize)),
                              wix=numpy.random.uniform(low=-numpy.sqrt(6. / (inputsize + hiddensize + hiddensize)),
                                                       high=numpy.sqrt(6. / (inputsize + hiddensize + hiddensize)),
                                                       size=(hiddensize, inputsize)),
                              bi=numpy.random.uniform(low=-numpy.sqrt(6. / (inputsize + hiddensize + hiddensize)),
                                                      high=numpy.sqrt(6. / (inputsize + hiddensize + hiddensize)),
                                                      size=(hiddensize, )),
                              wch=numpy.random.uniform(low=-numpy.sqrt(6. / (inputsize + hiddensize + hiddensize)),
                                                       high=numpy.sqrt(6. / (inputsize + hiddensize + hiddensize)),
                                                       size=(hiddensize, hiddensize)),
                              wcx=numpy.random.uniform(low=-numpy.sqrt(6. / (inputsize + hiddensize + hiddensize)),
                                                       high=numpy.sqrt(6. / (inputsize + hiddensize + hiddensize)),
                                                       size=(hiddensize, inputsize)),
                              bc=numpy.random.uniform(low=-numpy.sqrt(6. / (inputsize + hiddensize + hiddensize)),
                                                      high=numpy.sqrt(6. / (inputsize + hiddensize + hiddensize)),
                                                      size=(hiddensize, )),
                              woh=numpy.random.uniform(low=-numpy.sqrt(6. / (inputsize + hiddensize + hiddensize)),
                                                       high=numpy.sqrt(6. / (inputsize + hiddensize + hiddensize)),
                                                       size=(hiddensize, hiddensize)),
                              wox=numpy.random.uniform(low=-numpy.sqrt(6. / (inputsize + hiddensize + hiddensize)),
                                                       high=numpy.sqrt(6. / (inputsize + hiddensize + hiddensize)),
                                                       size=(hiddensize, inputsize)),
                              bo=numpy.random.uniform(low=-numpy.sqrt(6. / (inputsize + hiddensize + hiddensize)),
                                                      high=numpy.sqrt(6. / (inputsize + hiddensize + hiddensize)),
                                                      size=(hiddensize, )),
                              wy=numpy.random.uniform(low=-numpy.sqrt(6. / (hiddensize + outputsize)),
                                                      high=numpy.sqrt(6. / (hiddensize + outputsize)),
                                                      size=(outputsize, hiddensize)),
                              by=numpy.random.uniform(low=-numpy.sqrt(6. / (hiddensize + outputsize)),
                                                      high=numpy.sqrt(6. / (hiddensize + outputsize)),
                                                      size=(outputsize, )))
        self.layer = dict(initial_s=None, initial_h=None, input=None, 
                          f=None, f_activated=None,
                          i=None, i_activated=None,
                          c=None, c_activated=None,
                          o=None, o_activated=None,
                          s=None, s_activated=None,
                          h=None, output=None, probability=None)
        self.gradient = dict(wfh=None, wfx=None, bf=None,
                             wih=None, wix=None, bi=None,
                             wch=None, wcx=None, bc=None,
                             woh=None, wox=None, bo=None,
                             wy=None, by=None,
                             f=None, f_activated=None,
                             i=None, i_activated=None,
                             c=None, c_activated=None,
                             o=None, o_activated=None,
                             s=None, s_activated=None,
                             h=None, output=None)

    def forward(self, initial_s, initial_h, input):
        self.layer['initial_s'] = initial_s
        self.layer['initial_h'] = initial_h
        self.layer['input'] = input
        timesteps = len(input)
        self.layer['f'] = numpy.array([None] * timesteps)
        self.layer['i'] = numpy.array([None] * timesteps)
        self.layer['c'] = numpy.array([None] * timesteps)
        self.layer['o'] = numpy.array([None] * timesteps)
        self.layer['s'] = numpy.array([None] * timesteps)
        self.layer['f_activated'] = numpy.array([None] * timesteps)
        self.layer['i_activated'] = numpy.array([None] * timesteps)
        self.layer['c_activated'] = numpy.array([None] * timesteps)
        self.layer['o_activated'] = numpy.array([None] * timesteps)
        self.layer['s_activated'] = numpy.array([None] * timesteps)
        self.layer['h'] = numpy.array([None] * timesteps)
        for t in range(timesteps):
            if t == 0:
                self.layer['f'][t] = numpy.dot(self.parameter['wfh'], self.layer['initial_h']) + \
                                     numpy.dot(self.parameter['wfx'], self.layer['input'][t]) + \
                                     self.parameter['bf']
                self.layer['i'][t] = numpy.dot(self.parameter['wih'], self.layer['initial_h']) + \
                                     numpy.dot(self.parameter['wix'], self.layer['input'][t]) + \
                                     self.parameter['bi']
                self.layer['c'][t] = numpy.dot(self.parameter['wch'], self.layer['initial_h']) + \
                                     numpy.dot(self.parameter['wcx'], self.layer['input'][t]) + \
                                     self.parameter['bc']
                self.layer['o'][t] = numpy.dot(self.parameter['woh'], self.layer['initial_h']) + \
                                     numpy.dot(self.parameter['wox'], self.layer['input'][t]) + \
                                     self.parameter['bo']
            else:
                self.layer['f'][t] = numpy.dot(self.parameter['wfh'], self.layer['h'][t - 1]) + \
                                     numpy.dot(self.parameter['wfx'], self.layer['input'][t]) + \
                                     self.parameter['bf']
                self.layer['i'][t] = numpy.dot(self.parameter['wih'], self.layer['h'][t - 1]) + \
                                     numpy.dot(self.parameter['wix'], self.layer['input'][t]) + \
                                     self.parameter['bi']
                self.layer['c'][t] = numpy.dot(self.parameter['wch'], self.layer['h'][t - 1]) + \
                                     numpy.dot(self.parameter['wcx'], self.layer['input'][t]) + \
                                     self.parameter['bc']
                self.layer['o'][t] = numpy.dot(self.parameter['woh'], self.layer['h'][t - 1]) + \
                                     numpy.dot(self.parameter['wox'], self.layer['input'][t]) + \
                                     self.parameter['bo']
            self.layer['f_activated'][t] = function.sigmoid(self.layer['f'][t])
            self.layer['i_activated'][t] = function.sigmoid(self.layer['i'][t])
            self.layer['c_activated'][t] = function.tanh(self.layer['c'][t])
            self.layer['o_activated'][t] = function.sigmoid(self.layer['o'][t])
            if t == 0:
                self.layer['s'][t] = self.layer['f_activated'][t] * self.layer['initial_s'] + \
                                     self.layer['i_activated'][t] * self.layer['c_activated'][t]
            else:
                self.layer['s'][t] = self.layer['f_activated'][t] * self.layer['s'][t - 1] + \
                                     self.layer['i_activated'][t] * self.layer['c_activated'][t]
            self.layer['s_activated'][t] = function.tanh(self.layer['s'][t])
            self.layer['h'][t] = self.layer['o_activated'][t] * self.layer['s_activated'][t]
        self.layer['output'] = numpy.dot(self.parameter['wy'], self.layer['h'][-1]) + self.parameter['by']
        self.layer['probability'] = function.softmax(self.layer['output'])
        return self.layer['probability']

    def error(self, probability, label):
        return function.negative_log(probability, label)

    def backward(self, probability, label):
        self.gradient['output'] = function.d_negative_log(probability, label)
        timesteps = len(self.layer['input'])
        self.gradient['h'] = numpy.array([None] * timesteps)
        self.gradient['s_activated'] = numpy.array([None] * timesteps)
        self.gradient['o_activated'] = numpy.array([None] * timesteps)
        self.gradient['c_activated'] = numpy.array([None] * timesteps)
        self.gradient['i_activated'] = numpy.array([None] * timesteps)
        self.gradient['f_activated'] = numpy.array([None] * timesteps)
        self.gradient['s'] = numpy.array([None] * timesteps)
        self.gradient['o'] = numpy.array([None] * timesteps)
        self.gradient['c'] = numpy.array([None] * timesteps)
        self.gradient['i'] = numpy.array([None] * timesteps)
        self.gradient['f'] = numpy.array([None] * timesteps)
        for t in range(timesteps - 1, -1, -1):
            if t == timesteps - 1:
                self.gradient['h'][t] = numpy.dot(self.gradient['output'], self.parameter['wy'])
                self.gradient['s_activated'][t] = self.gradient['h'][t] * self.layer['o_activated'][t]
                self.gradient['s'][t] = self.gradient['s_activated'][t] * function.d_tanh(self.layer['s'][t])
            else:
                self.gradient['h'][t] = numpy.dot(self.gradient['o'][t + 1], self.parameter['woh']) + \
                                        numpy.dot(self.gradient['c'][t + 1], self.parameter['wch']) + \
                                        numpy.dot(self.gradient['i'][t + 1], self.parameter['wih']) + \
                                        numpy.dot(self.gradient['f'][t + 1], self.parameter['wfh'])
                self.gradient['s_activated'][t] = self.gradient['h'][t] * self.layer['o_activated'][t]
                self.gradient['s'][t] = self.gradient['s_activated'][t] * function.d_tanh(self.layer['s'][t]) + \
                                        self.gradient['s'][t + 1] * self.layer['f_activated'][t + 1]
            self.gradient['o_activated'][t] = self.gradient['h'][t] * self.layer['s_activated'][t]
            self.gradient['c_activated'][t] = self.gradient['s'][t] * self.layer['i_activated'][t]
            self.gradient['i_activated'][t] = self.gradient['s'][t] * self.layer['c_activated'][t]
            if t > 0:
                self.gradient['f_activated'][t] = self.gradient['s'][t] * self.layer['s'][t - 1]
            else:
                self.gradient['f_activated'][t] = self.gradient['s'][t] * self.layer['initial_s']
            self.gradient['o'][t] = self.gradient['o_activated'][t] * function.d_sigmoid(self.layer['o'][t])
            self.gradient['c'][t] = self.gradient['c_activated'][t] * function.d_tanh(self.layer['c'][t])
            self.gradient['i'][t] = self.gradient['i_activated'][t] * function.d_sigmoid(self.layer['i'][t])
            self.gradient['f'][t] = self.gradient['f_activated'][t] * function.d_sigmoid(self.layer['f'][t])
        self.gradient['by'] = self.gradient['output']
        self.gradient['wy'] = self.gradient['output'].reshape(-1, 1) * self.layer['h'][-1]
        self.gradient['bo'] = numpy.zeros_like(self.parameter['bo'])
        self.gradient['bc'] = numpy.zeros_like(self.parameter['bc'])
        self.gradient['bi'] = numpy.zeros_like(self.parameter['bi'])
        self.gradient['bf'] = numpy.zeros_like(self.parameter['bf'])
        self.gradient['wox'] = numpy.zeros_like(self.parameter['wox'])
        self.gradient['wcx'] = numpy.zeros_like(self.parameter['wcx'])
        self.gradient['wix'] = numpy.zeros_like(self.parameter['wix'])
        self.gradient['wfx'] = numpy.zeros_like(self.parameter['wfx'])
        self.gradient['woh'] = numpy.zeros_like(self.parameter['woh'])
        self.gradient['wch'] = numpy.zeros_like(self.parameter['wch'])
        self.gradient['wih'] = numpy.zeros_like(self.parameter['wih'])
        self.gradient['wfh'] = numpy.zeros_like(self.parameter['wfh'])
        for t in range(timesteps):
            self.gradient['bo'] += self.gradient['o'][t]
            self.gradient['bc'] += self.gradient['c'][t]
            self.gradient['bi'] += self.gradient['i'][t]
            self.gradient['bf'] += self.gradient['f'][t]
            self.gradient['wox'] += self.gradient['o'][t].reshape(-1, 1) * self.layer['input'][t]
            self.gradient['wcx'] += self.gradient['c'][t].reshape(-1, 1) * self.layer['input'][t]
            self.gradient['wix'] += self.gradient['i'][t].reshape(-1, 1) * self.layer['input'][t]
            self.gradient['wfx'] += self.gradient['f'][t].reshape(-1, 1) * self.layer['input'][t]
            if t > 0:
                self.gradient['woh'] += self.gradient['o'][t].reshape(-1, 1) * self.layer['h'][t - 1]
                self.gradient['wch'] += self.gradient['c'][t].reshape(-1, 1) * self.layer['h'][t - 1]
                self.gradient['wih'] += self.gradient['i'][t].reshape(-1, 1) * self.layer['h'][t - 1]
                self.gradient['wfh'] += self.gradient['f'][t].reshape(-1, 1) * self.layer['h'][t - 1]
            else:
                self.gradient['woh'] += self.gradient['o'][t].reshape(-1, 1) * self.layer['initial_h']
                self.gradient['wch'] += self.gradient['c'][t].reshape(-1, 1) * self.layer['initial_h']
                self.gradient['wih'] += self.gradient['i'][t].reshape(-1, 1) * self.layer['initial_h']
                self.gradient['wfh'] += self.gradient['f'][t].reshape(-1, 1) * self.layer['initial_h']
        return self.gradient
