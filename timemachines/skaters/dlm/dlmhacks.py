from timemachines.skaters.dlm.dlmunivariate import using_dlm, autoReg, dlm
if using_dlm:

    class fixedAutoReg(autoReg):
        # A hack to make DLM auto work, the version in PyPI at time of writing throws error trying to append

        def __init__(self,*arg,**kwargs):
            super().__init__(*arg,**kwargs)

        def appendNewData(self, data):
            """ AutoReg append new data automatically with the main time series. Nothing
            needs to be done here.
            """
            return


    class quietDlm(dlm):
        # Same as DLM but without priting

        def __init__(self,*arg,**kwargs):
            super().__init__(*arg,**kwargs)

        # Append new data or features to the dlm
        def append(self, data, component='main'):
            """ Append the new data to the main data or the components (new feature data)

            Args:
                data: the new data
                component: the name of which the new data to be added to.\n
                           'main': the main time series data\n
                           other omponent name: add new feature data to other
                           component.

            """
            # initialize the model to ease the modification
            if not self.initialized:
                self._initialize()

            # if we are adding new data to the time series
            if component == 'main':
                # add the data to the self.data
                self.data.extend(list(data))

                # update the length
                self.n += len(data)
                self.result._appendResult(len(data))

                # update the automatic components as well
                for component in self.builder.automaticComponents:
                    comp = self.builder.automaticComponents[component]
                    comp.appendNewData(data)

            # if we are adding new data to the features of dynamic components
            elif component in self.builder.dynamicComponents:
                comp = self.builder.dynamicComponents[component]
                comp.appendNewData(data)

            else:
                raise NameError('Such dynamic component does not exist.')
