#include <TChain.h>
#include <TTreeReader.h>
#include <TTreeReaderValue.h>
#include <TTreeReaderArray.h>
#include <iostream>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <cmath>

#pragma pack(push, 1)
struct Row {
  uint16_t event_id;
  uint8_t  module_id;
  uint16_t strip_id;
  int16_t  adc[6];
};
#pragma pack(pop)

void dataReady_inference_ML(Int_t run_number_int = 11590,
                        Int_t analyzeOnTrackOrNot = 0,
                        Int_t nevents_display = 10,
                        const TString output_file_name = "decoding_digitized_data_GEP_16000_25uA.pdf")
{
 
  std::ofstream dataFile("data_for_inference_ML_withoutROIcut.txt");
  dataFile << std::setprecision(12);

  TChain* tchain_T = new TChain("T");
  tchain_T->Add("/volatile/halla/sbs/bhasitha/Tracking_ML/GEM_Decoder_EVALUATION/filtered_replayed.root");
  
  TTreeReader r(tchain_T);

//   TTreeReaderValue<Int_t> nstrips(r, "Harm.FT.dighit.nstrips");
  TTreeReaderArray<double> module(r,   "sbs.gemFT.hit.module");
  TTreeReaderArray<double> adc(r,      "sbs.gemFT.m0.strip.ADCsamples");
  TTreeReaderArray<double> strip(r,    "sbs.gemFT.m0.strip.istrip");
  TTreeReaderArray<double> isU(r,    "sbs.gemFT.m0.strip.IsU");                                  
  TTreeReaderArray<double> isV(r,    "sbs.gemFT.m0.strip.IsV");

  TTreeReaderArray<double> adc_good(r, "sbs.gemFT.m0.strip.ADCsamples_goodADC");
  TTreeReaderArray<double> strip_good(r,    "sbs.gemFT.m0.strip.istrip_goodADC");
  TTreeReaderArray<double> isU_good(r,    "sbs.gemFT.m0.strip.IsU_goodADC");                                  
  TTreeReaderArray<double> isV_good(r,    "sbs.gemFT.m0.strip.IsV_goodADC");


  TTreeReaderValue<double> inmod(r,   "sbs.gemFT.m0.roi.inmod");
  TTreeReaderValue<double> ustrip_max(r,   "sbs.gemFT.m0.roi.ustrip_max");
  TTreeReaderValue<double> ustrip_min(r,   "sbs.gemFT.m0.roi.ustrip_min");
  TTreeReaderValue<double> vstrip_max(r,   "sbs.gemFT.m0.roi.vstrip_max");
  TTreeReaderValue<double> vstrip_min(r,   "sbs.gemFT.m0.roi.vstrip_min");

  constexpr int ntimesamples = 6;

  Long64_t nevent = 0;



  // while (r.Next() && nevent < 100000) {
  while (r.Next()) {
 
    const size_t Nadc    = adc.GetSize();
    const size_t Nstrip  = strip.GetSize();
    const size_t Nmod    = module.GetSize();
    // const size_t Nadc_good   = adc_good.GetSize();

    // 2) number of strips implied by adc
    const size_t nstrips_adc = Nadc / ntimesamples;
    // const size_t nstrips_adc_good = Nadc_good / ntimesamples;

    // 3) effective strip count = min of all per-strip vectors + adc-derived
    // const size_t nstrip_eff = std::min({ nstrips_from_adc, Nstrip, Nmod, Ngood });
    const size_t nstrip_eff = nstrips_adc; //std::min({ nstrips_from_adc, Nstrip, Nmod, Ngood });
    // const size_t nstrip_eff_good = nstrips_adc_good;


    // ======================================================================== remove too large ROIs  ======================================================================== 



    // variables to track if we found a match for the current strip
    bool found_strip_ID = false;
    std::vector<double> matched_row;
    double readout_write = -1000;

    // Now loop over all strips in the current event and try to find matches in the good ADC array
    for (size_t istrip = 0; istrip < nstrip_eff; ++istrip) {

      // if (!(module[istrip * ntimesamples] == 0 || module[istrip * ntimesamples] == 1)) continue;
      // if (module[istrip * ntimesamples] != 0 ) continue;

      // extract the 6 ADC samples for this strip
      const size_t s0 = istrip * ntimesamples;
      double a[6] = {
        adc[s0+0], adc[s0+1], adc[s0+2],
        adc[s0+3], adc[s0+4], adc[s0+5]
      };

      readout_write = -1000;
      // determine readout_write type
      if (isU[istrip] == 1) {
        readout_write = 0;
      } else if (isV[istrip] == 1) {
        readout_write = 1;
      } 
    
      // ======================================================================== CUTS ========================================================================
      // strip outside ROI or invalid readout_write
      if (readout_write == -1000) { continue; }
      if (readout_write == 0 && (strip[istrip] < *ustrip_min || strip[istrip] > *ustrip_max)) { continue;}
      if (readout_write == 1 && (strip[istrip] < *vstrip_min || strip[istrip] > *vstrip_max)) { continue;}

\
      
    dataFile 
            << nevent << " "
            << readout_write << " "
            << strip[istrip] << " "
            << a[0] << " " << a[1] << " " << a[2] << " "
            << a[3] << " " << a[4] << " " << a[5] << "\n";

    //   found_strip_ID = false;
    }

    ++nevent;
  }
  cout << "Total events processed: " << nevent << std::endl;
  dataFile.close();
}
