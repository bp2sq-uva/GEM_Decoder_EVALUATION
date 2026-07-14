#include <TChain.h>
#include <TTreeReader.h>
#include <TTreeReaderValue.h>
#include <TTreeReaderArray.h>
#include <TSystem.h>

#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdint>

#pragma pack(push, 1)
struct Row {
  uint16_t event_id;
  uint8_t  module_id;
  uint16_t strip_id;
  int16_t  adc[6];
};
#pragma pack(pop)

void ground_truth_generator(
    // Int_t run_number_int = 11590,
    // Int_t analyzeOnTrackOrNot = 0,
    // Int_t nevents_display = 10,
    // const TString output_file_name = "decoding_digitized_data_GEP_16000_25uA.pdf"
)
{
  const TString input_root_file =
    "/volatile/halla/sbs/bhasitha/Tracking_ML/GEM_Decoder_EVALUATION/filtered_replayed_withoutROIcut.root";

  const TString output_txt_file =
    "Scratch/Truth_info/groundtruth_withoutROIcut.txt";

  // Make sure output directory exists.
  gSystem->mkdir("Scratch/Truth_info", kTRUE);

  std::ofstream dataFile(output_txt_file.Data());
  dataFile << std::setprecision(12);

  if (!dataFile.is_open()) {
    std::cerr << "ERROR: Could not open output file: "
              << output_txt_file << std::endl;
    return;
  }

  TChain* tchain_T = new TChain("T");
  tchain_T->Add(input_root_file);

  if (tchain_T->GetEntries() == 0) {
    std::cerr << "ERROR: No entries found in input file: "
              << input_root_file << std::endl;
    dataFile.close();
    return;
  }

  TTreeReader r(tchain_T);

  // Regular strip branches. These are used only to verify that the goodADC
  // strip also exists in the regular strip list with the same U/V side.
  TTreeReaderArray<double> strip(r, "sbs.gemFT.m0.strip.istrip");
  TTreeReaderArray<double> isU(r,   "sbs.gemFT.m0.strip.IsU");
  TTreeReaderArray<double> isV(r,   "sbs.gemFT.m0.strip.IsV");

  // goodADC branches to write.
  TTreeReaderArray<double> adc_good(r,   "sbs.gemFT.m0.strip.ADCsamples_goodADC");
  TTreeReaderArray<double> strip_good(r, "sbs.gemFT.m0.strip.istrip_goodADC");
  TTreeReaderArray<double> isU_good(r,   "sbs.gemFT.m0.strip.IsU_goodADC");
  TTreeReaderArray<double> isV_good(r,   "sbs.gemFT.m0.strip.IsV_goodADC");

  // ROI branches.
  TTreeReaderValue<double> inmod(r,       "sbs.gemFT.m0.roi.inmod");
  TTreeReaderValue<double> ustrip_max(r,  "sbs.gemFT.m0.roi.ustrip_max");
  TTreeReaderValue<double> ustrip_min(r,  "sbs.gemFT.m0.roi.ustrip_min");
  TTreeReaderValue<double> vstrip_max(r,  "sbs.gemFT.m0.roi.vstrip_max");
  TTreeReaderValue<double> vstrip_min(r,  "sbs.gemFT.m0.roi.vstrip_min");

  constexpr int ntimesamples = 6;

  Long64_t nevent = 0;
  Long64_t n_written = 0;
  Long64_t n_skip_inmod = 0;
  Long64_t n_skip_bad_readout = 0;
  Long64_t n_skip_outside_roi = 0;
  Long64_t n_skip_no_regular_match = 0;
  Long64_t n_skip_bad_adc_size = 0;

  while (r.Next()) {

    if (*inmod != 1) {
      n_skip_inmod++;
      nevent++;
      continue;
    }

    const size_t nstrip_eff = std::min({
      strip.GetSize(),
      isU.GetSize(),
      isV.GetSize()
    });

    const size_t nstrip_eff_good = std::min({
      strip_good.GetSize(),
      isU_good.GetSize(),
      isV_good.GetSize(),
      adc_good.GetSize() / ntimesamples
    });

    if (strip_good.GetSize() > adc_good.GetSize() / ntimesamples) {
      n_skip_bad_adc_size++;
    }

    for (size_t istrip_good = 0; istrip_good < nstrip_eff_good; ++istrip_good) {

      int readout_write = -1;

      if (isU_good[istrip_good] == 1) {
        readout_write = 0;
      }
      else if (isV_good[istrip_good] == 1) {
        readout_write = 1;
      }
      else {
        n_skip_bad_readout++;
        continue;
      }

      const double good_strip_id = strip_good[istrip_good];

      // ROI cut on goodADC strip.
      if (readout_write == 0 &&
          (good_strip_id < *ustrip_min || good_strip_id > *ustrip_max)) {
        n_skip_outside_roi++;
        std::cout << "Skipping goodADC strip outside ROI: "
                  << "event_id=" << nevent
                  << ", readout=" << readout_write
                  << ", strip_id=" << good_strip_id
                  << ", ustrip_min=" << *ustrip_min
                  << ", ustrip_max=" << *ustrip_max
                  << std::endl;
        continue;
      }

      if (readout_write == 1 &&
          (good_strip_id < *vstrip_min || good_strip_id > *vstrip_max)) {
        n_skip_outside_roi++;
        std::cout << "Skipping goodADC strip outside ROI: "
                  << "event_id=" << nevent
                  << ", readout=" << readout_write
                  << ", strip_id=" << good_strip_id
                  << ", vstrip_min=" << *vstrip_min
                  << ", vstrip_max=" << *vstrip_max
                  << std::endl;
        continue;
      }

      // Require this goodADC strip to exist in the regular strip list
      // with the same U/V side.
      bool found_regular_match = false;

      for (size_t i = 0; i < nstrip_eff; ++i) {

        const bool same_strip =
          std::llround(strip[i]) == std::llround(good_strip_id);

        const bool same_readout =
          (readout_write == 0 && isU[i] == 1) ||
          (readout_write == 1 && isV[i] == 1);

        if (same_strip && same_readout) {
          found_regular_match = true;
          break;
        }
      }

      if (!found_regular_match) {
        n_skip_no_regular_match++;
        std::cout << "Skipping goodADC strip with no matching regular strip: "
                  << "event_id=" << nevent
                  << ", readout=" << readout_write
                  << ", strip_id=" << good_strip_id
                  << std::endl;
      }

      const size_t s0_good = istrip_good * ntimesamples;

      dataFile
        << nevent << " "
        << readout_write << " "
        << std::llround(good_strip_id) << " "
        << adc_good[s0_good + 0] << " "
        << adc_good[s0_good + 1] << " "
        << adc_good[s0_good + 2] << " "
        << adc_good[s0_good + 3] << " "
        << adc_good[s0_good + 4] << " "
        << adc_good[s0_good + 5] << "\n";

      n_written++;
    }

    nevent++;
  }

  dataFile.close();

  std::cout << "\nFinished writing goodADC truth file.\n";
  std::cout << "Input ROOT file:                  " << input_root_file << "\n";
  std::cout << "Output text file:                 " << output_txt_file << "\n";
  std::cout << "Total events processed:           " << nevent << "\n";
  std::cout << "Rows written:                     " << n_written << "\n";
  std::cout << "Skipped events, inmod != 1:       " << n_skip_inmod << "\n";
  std::cout << "Skipped strips, bad readout:      " << n_skip_bad_readout << "\n";
  std::cout << "Skipped strips, outside ROI:      " << n_skip_outside_roi << "\n";
  std::cout << "Skipped strips, no regular match: " << n_skip_no_regular_match << "\n";
  std::cout << "Events/strips with bad ADC size:  " << n_skip_bad_adc_size << "\n";
}