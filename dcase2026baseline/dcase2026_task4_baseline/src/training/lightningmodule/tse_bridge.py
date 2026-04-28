from .tse import TSELightning


class TSEBridgeLightning(TSELightning):
    """Opt-in TSE lightning module that forwards USS bridge conditions.

    Existing TSELightning remains unchanged. This class extends only the input
    dictionary with optional bridge features loaded by BridgeTSEDataset or
    BridgeEstimatedEnrollmentTSEDataset.
    """

    def _get_input_dict(self, batch_data_dict):
        input_dict = super()._get_input_dict(batch_data_dict)
        for key in ("bridge_condition", "tse_condition", "proposal_condition"):
            if key in batch_data_dict:
                input_dict[key] = batch_data_dict[key]
        return input_dict
