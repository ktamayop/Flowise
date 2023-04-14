import PropTypes from 'prop-types'
import { Handle, Position, useUpdateNodeInternals } from 'reactflow'
import { useEffect, useRef, useState, useContext } from 'react'

// material-ui
import { useTheme, styled } from '@mui/material/styles'
import { Box, Typography, Tooltip } from '@mui/material'
import { tooltipClasses } from '@mui/material/Tooltip'
import { Dropdown } from 'ui-component/dropdown/Dropdown'
import { Input } from 'ui-component/input/Input'
import { File } from 'ui-component/file/File'
import { flowContext } from 'store/context/ReactFlowContext'
import { isValidConnection } from 'utils/genericHelper'

const CustomWidthTooltip = styled(({ className, ...props }) => <Tooltip {...props} classes={{ popper: className }} />)({
    [`& .${tooltipClasses.tooltip}`]: {
        maxWidth: 500
    }
})

// ===========================|| NodeInputHandler ||=========================== //

const def load_checkpoint(model, folder):
    """
    Loads a PyTorch model from a sharded checkpoint.

    Args:
        model (torch.nn.Module): The model to which the checkpoint will be loaded.
        folder (str): The path to the folder containing checkpoint data.

    Returns:
        A named tuple with missing_keys and unexpected_keys fields
        - missing_keys is a list of str containing the missing keys
        - unexpected_keys is a list of str containing the unexpected keys
    """
    index_file = os.path.join(folder, WEIGHTS_INDEX_NAME)
    try:
        with open(index_file, "r", encoding="utf-8") as f:
            index = torch.load(f)
    except FileNotFoundError as e:
        raise OSError(f"Could not find checkpoint index file ({WEIGHTS_INDEX_NAME}) in {folder}: {e}")

    shard_files = set(index["weight_map"].values())
    loaded_keys = index["weight_map"].keys()

    model_keys = model.state_dict().keys()
    missing_keys = [k for k in model_keys if k not in loaded_keys]
    unexpected_keys = [k for k in loaded_keys if k not in model_keys]

    for shard_file in shard_files:
        state_dict = torch.load(os.path.join(folder, shard_file))
        model.load_state_dict(state_dict, strict=False)

        del state_dict
        gc.collect()

    return torch.nn.modules.module._IncompatibleKeys(missing_keys, unexpected_keys)

NodeInputHandler.propTypes = {
    inputAnchor: PropTypes.object,
    inputParam: PropTypes.object,
    data: PropTypes.object,
    disabled: PropTypes.bool
}

export default NodeInputHandler
