#![allow(non_snake_case)]
use autocxx::prelude::*;
use autocxx::subclass::*;
use std::ffi::CString;
use std::pin::Pin;
use td_rs_base::{param::ParameterManager, OperatorInputs, InfoDat, InfoChop};

use crate::TopOutput;
// use crate::mode::cpu::{TopCpuInput, TopCpuOutput};
use crate::Top;

include_cpp! {
    #include "TOP_CPlusPlusBase.h"
    #include "RustTopPlugin.h"
    safety!(unsafe)
    extern_cpp_type!("TD::OP_ParameterManager", td_rs_base::cxx::OP_ParameterManager)
    extern_cpp_type!("TD::OP_String", td_rs_base::cxx::OP_String)
    extern_cpp_type!("TD::OP_InfoDATSize", td_rs_base::cxx::OP_InfoDATSize)
    extern_cpp_type!("TD::OP_InfoCHOPChan", td_rs_base::cxx::OP_InfoCHOPChan)
    extern_cpp_type!("TD::OP_Inputs", td_rs_base::cxx::OP_Inputs)
    extern_cpp_type!("TD::OP_TOPInput", td_rs_base::cxx::OP_TOPInput)
    extern_cpp_type!("TD::OP_TOPInputDownloadOptions", td_rs_base::cxx::OP_TOPInputDownloadOptions)
    extern_cpp_type!("TD::OP_CPUMemPixelType", td_rs_base::cxx::OP_CPUMemPixelType)
    pod!("TD::OP_CPUMemPixelType")
    generate_pod!("TD::TOP_GeneralInfo")
    generate_pod!("TD::TOP_PluginInfo")
}

pub use ffi::TD::*;
pub use ffi::*;
pub use td_rs_base::cxx::setString;

extern "C" {
    fn top_new_impl() -> Box<dyn Top>;
}

#[subclass(superclass("RustTopPlugin"))]
pub struct RustTopPluginImpl {
    inner: Box<dyn Top>,
}

impl Default for RustTopPluginImpl {
    fn default() -> Self {
        unsafe {
            Self {
                inner: top_new_impl(),
                cpp_peer: Default::default(),
            }
        }
    }
}

#[no_mangle]
extern "C" fn top_new() -> *mut RustTopPluginImplCpp {
    RustTopPluginImpl::default_cpp_owned().into_raw()
}

impl RustTopPlugin_methods for RustTopPluginImpl {
    fn getGeneralInfo(&mut self, mut info: Pin<&mut TOP_GeneralInfo>, inputs: &OP_Inputs) {
        let input = OperatorInputs::new(inputs);
        let gen_info = self.inner.general_info(&input);
        info.cookEveryFrame = gen_info.cook_every_frame;
        info.cookEveryFrameIfAsked = gen_info.cook_every_frame_if_asked;
        info.inputSizeIndex = gen_info.input_size_index;
    }

    // TOP_OutputFormatSpecs &output_specs, const OP_Inputs &inputs, TOP_Context &context
    fn execute(&mut self, output: Pin<&mut TOP_Output>, inputs: &OP_Inputs) {
        let input = OperatorInputs::new(inputs);
        let output = TopOutput::new(output);
        self.inner.execute(output, &input);
    }

    fn getNumInfoCHOPChans(&mut self) -> i32 {
        InfoChop::size(&self.inner) as i32
    }

    fn getInfoCHOPChan(&mut self, index: i32, name: Pin<&mut OP_String>, mut value: Pin<&mut f32>) {
        let (info_name, info_value) = InfoChop::channel(&self.inner, index as usize);
        unsafe {
            let new_string = CString::new(info_name.as_str()).unwrap();
            let new_string_ptr = new_string.as_ptr();
            name.setString(new_string_ptr);
        }
        value.set(info_value);
    }

    fn getInfoDATSize(&mut self, mut info: Pin<&mut OP_InfoDATSize>) -> bool {
        let (rows, cols) = InfoDat::size(&self.inner);
        if rows == 0 && cols == 0 {
            false
        } else {
            info.rows = rows as i32;
            info.cols = cols as i32;
            true
        }
    }

    fn getInfoDATEntry(&mut self, index: i32, entryIndex: i32, entry: Pin<&mut OP_String>) {
        let entry_str = InfoDat::entry(&self.inner, index as usize, entryIndex as usize);
        unsafe {
            let new_string = CString::new(entry_str.as_str()).unwrap();
            let new_string_ptr = new_string.as_ptr();
            entry.setString(new_string_ptr);
        }
    }

    fn getWarningString(&mut self, warning: Pin<&mut OP_String>) {
        unsafe {
            let new_string = CString::new(self.inner.warning()).unwrap();
            let new_string_ptr = new_string.as_ptr();
            warning.setString(new_string_ptr);
        }
    }

    fn getErrorString(&mut self, error: Pin<&mut OP_String>) {
        unsafe {
            let new_string = CString::new(self.inner.error()).unwrap();
            let new_string_ptr = new_string.as_ptr();
            error.setString(new_string_ptr);
        }
    }

    fn getInfoPopupString(&mut self, info: Pin<&mut OP_String>) {
        unsafe {
            let new_string = CString::new(self.inner.info()).unwrap();
            let new_string_ptr = new_string.as_ptr();
            info.setString(new_string_ptr);
        }
    }

    fn setupParameters(&mut self, manager: Pin<&mut OP_ParameterManager>) {
        let params = self.inner.params_mut();
        if let Some(mut params) = params {
            let mut manager = ParameterManager::new(manager);
            params.register(&mut manager);
        }
    }

    unsafe fn pulsePressed(&mut self, name: *const std::ffi::c_char) {
        self.inner
            .pulse_pressed(std::ffi::CStr::from_ptr(name).to_str().unwrap());
    }
}