use crate::datareader::DataReader;
use crate::Extractor;
use ncnn_bind::*;
use std::ffi::CString;

pub struct Net {
    ptr: ncnn_net_t,
}

unsafe impl Send for Net {}

impl Net {
    pub fn new() -> Net {
        Net {
            ptr: unsafe { ncnn_net_create() },
        }
    }

    pub fn set_option(&mut self, opt: &crate::option::Option) {
        unsafe {
            ncnn_net_set_option(self.ptr, opt.ptr());
        }
    }

    pub fn load_param(&mut self, path: &str) -> anyhow::Result<()> {
        let c_str = CString::new(path).unwrap();
        if unsafe { ncnn_net_load_param(self.ptr, c_str.as_ptr()) } != 0 {
            anyhow::bail!("Error loading params {}", path);
        } else {
            Ok(())
        }
    }

    pub fn load_param_from_slice(&mut self, param: &[u8]) -> anyhow::Result<()> {
        // TODO: ensure the slice is loaded correctly then switch to direct method
        //let param_ptr = param.as_ptr() as *const c_char;
        //if unsafe { ncnn_net_load_param_memory(self.ptr, param_ptr) } != 0 {
        //anyhow::bail!("Error loading params");
        //} else {
        //Ok(())
        //}
        let temp_file_name = format!("/tmp/{}.param", rand::random::<u64>().to_string());
        std::fs::write(&temp_file_name, param)?;
        let res = self.load_param(&temp_file_name);
        std::fs::remove_file(&temp_file_name)?;
        res
    }

    pub fn load_model(&mut self, path: &str) -> anyhow::Result<()> {
        let c_str = CString::new(path).unwrap();
        if unsafe { ncnn_net_load_model(self.ptr, c_str.as_ptr()) } != 0 {
            anyhow::bail!("Error loading model {}", path);
        } else {
            Ok(())
        }
    }

    pub fn load_model_from_slice(&mut self, model: &[u8]) -> anyhow::Result<()> {
        // TODO: fix this to use direct method
        //let model_ptr = model.as_ptr() as *const c_uchar;
        //if unsafe { ncnn_net_load_model_memory(self.ptr, model_ptr) } != 0 {
        //anyhow::bail!("Error loading model");
        //} else {
        //Ok(())
        //}
        let temp_file_name = format!("/tmp/{}.model", rand::random::<u64>().to_string());
        std::fs::write(&temp_file_name, model)?;
        let res = self.load_model(&temp_file_name);
        std::fs::remove_file(&temp_file_name)?;
        res
    }

    pub fn load_model_datareader(&mut self, dr: &DataReader) -> anyhow::Result<()> {
        if unsafe { ncnn_net_load_model_datareader(self.ptr, dr.ptr()) } != 0 {
            anyhow::bail!("Error loading model from datareader");
        } else {
            Ok(())
        }
    }

    pub fn create_extractor(&mut self) -> Extractor<'_> {
        let ptr;
        unsafe {
            ptr = ncnn_extractor_create(self.ptr);
        }
        Extractor::from_ptr(ptr)
    }
}

impl Drop for Net {
    fn drop(&mut self) {
        unsafe {
            ncnn_net_destroy(self.ptr);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn is_send<T: Send>() -> bool {
        true
    }
    fn is_sync<T: Sync>() -> bool {
        true
    }

    #[test]
    fn load_not_exist_model() {
        let mut net = Net::new();
        net.load_param("not_exist.param")
            .expect_err("Expected param to be not found");
    }

    #[test]
    fn check_sync_send() {
        assert!(is_send::<Net>());
        //assert!(is_sync::<Net>());
    }
}
