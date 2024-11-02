/* Shared Use License: This file is owned by Derivative Inc. (Derivative)
 * and can only be used, and/or modified for use, in conjunction with
 * Derivative's TouchDesigner software, and only if you are a licensee who has
 * accepted Derivative's TouchDesigner license or assignment agreement
 * (which also govern the use of this file). You may share or redistribute
 * a modified version of this file provided the following conditions are met:
 *
 * 1. The shared file or redistribution must retain the information set out
 * above and this list of conditions.
 * 2. Derivative's name (Derivative Inc.) or its trademarks may not be used
 * to endorse or promote products derived from this file without specific
 * prior written permission from Derivative.
 */

/*
* Produced by:
*
* 				Derivative Inc
*				401 Richmond Street West, Unit 386
*				Toronto, Ontario
*				Canada   M5V 3A8
*				416-591-3555
*
* NAME:				DAT_CPlusPlusBase.h
*
*
*	Do not edit this file directly!
*	Make a subclass of DAT_CPlusPlusBase instead, and add your own
*	data/functions.

*	Derivative Developers:: Make sure the virtual function order
*	stays the same, otherwise changes won't be backwards compatible
*/
// #pragma once

#ifndef __DAT_CPlusPlusBase__
#define __DAT_CPlusPlusBase__

#include "CPlusPlus_Common.h"
#include <assert.h>

namespace TD {

#pragma pack(push, 8)

// Define for the current API version that this sample code is made for.
// To upgrade to a newer version, replace the files
// DAT_CPlusPlusBase.h
// CPlusPlus_Common.h
// from the samples folder in a newer TouchDesigner installation.
// You may need to upgrade your plugin code in that case, to match
// the new API requirements
const int DATCPlusPlusAPIVersion = 3;

class DAT_PluginInfo {
public:
  int32_t apiVersion = 0;

  int32_t reserved[100];

  // Information used to describe this plugin as a custom OP.
  OP_CustomOPInfo customOPInfo;

  int32_t reserved2[20];
};

class DAT_GeneralInfo {
public:
  // Set this to true if you want the DAT to cook every frame, even
  // if none of it's inputs/parameters are changing.
  // This is generally useful for cases where the node is outputting to
  // something external to TouchDesigner, such as a network socket or device.
  // It ensures the node cooks every if nothing inside the network is
  // using/viewing the output of this node. Important: If the node may not be
  // viewed/used by other nodes in the file, such as a TCP network output node
  // that isn't viewed in perform mode, you should set cookOnStart = true in
  // OP_CustomOPInfo. That will ensure cooking is kick-started for this node.
  // Note that this fix only works for Custom Operators, not
  // cases where the .dll is loaded into CPlusPlus DAT.
  // DEFAULT: false
  bool cookEveryFrame;

  // Set this to true if you want the DAT to cook every frame, but only
  // if someone asks for it to cook. So if nobody is using the output from
  // the DAT, it won't cook. This is difereent from 'cookEveryFrame'
  // since that will cause it to cook every frame no matter what.
  // DEFAULT: false
  bool cookEveryFrameIfAsked;

private:
  int32_t reserved[20];
};

enum class DAT_OutDataType {
  Table = 0,
  Text,
};

class DAT_Output {
public:
  DAT_Output() {}

  ~DAT_Output() {}

  // Set the type of output data, call this function at the very start to
  // specify whether a Table or Text data will be output.
  virtual void setOutputDataType(DAT_OutDataType type) = 0;

  virtual DAT_OutDataType getOutputDataType() = 0;

  // If the type of out data is Table, set the number of rows and columns.
  virtual void setTableSize(const int32_t rows, const int32_t cols) = 0;

  virtual void getTableSize(int32_t *rows, int32_t *cols) const = 0;

  // If the type of out data is set to Text,
  // Set the whole text by calling this function. str must be UTF-8 encoded.
  // returns false if null argument,
  // or if str is contains invalid UTF-8 bytes.
  virtual bool setText(const char *str) = 0;

  // Find the row/col index with a given name. name must be UTF-8 encoded.
  // The hintRowIndex/hintColIndex, if given and in range, will be
  // checked first to see if that row/col is a match.
  // This can make the searching faster if the row/col headers don't change
  // often. Returns -1 if it cannot find the row or if rowName isn't valid
  // UTF-8.
  virtual int32_t findRow(const char *rowName,
                          int32_t hint32_tRowIndex = -1) = 0;
  virtual int32_t findCol(const char *colName, int32_t hintColIndex = -1) = 0;

  // Set the string data for each cell of the table specified by a row and
  // column index, Returns false if such cell doesn't exists, or if str isn't
  // valid UTF-8.
  virtual bool setCellString(int32_t row, int32_t col, const char *str) = 0;

  // Set the int data for each cell, similar to the setCellString() but sets Int
  // values.
  virtual bool setCellInt(int32_t row, int32_t col, int32_t value) = 0;

  // Set the data for each cell, similar to the setCellString() but sets Double
  // values.
  virtual bool setCellDouble(int32_t row, int32_t col, double value) = 0;

  // Get the string cell data at a row and column index.
  // Returns null if the cell/table doesn't exist.
  // The memory the pointer points to is valid until the next call to
  // a function that changes the tabel (setCell*, setTableSize etc.)
  // or the end of the ::execute function.
  virtual const char *getCellString(int32_t row, int32_t col) const = 0;

  // Get the int32_t cell data with a row and column index,
  // returns false if it cannot find the cell, or invalid argument
  virtual bool getCellInt(int32_t row, int32_t col, int32_t *res) const = 0;

  // Get the double cell data with a row and column index,
  // returns false if it cannot find the cell, or invalid argument
  virtual bool getCellDouble(int32_t row, int32_t col, double *res) const = 0;

private:
  int32_t reserved[20];
};

/*** DO NOT EDIT THIS CLASS, MAKE A SUBCLASS OF IT INSTEAD ***/
class DAT_CPlusPlusBase {
protected:
  DAT_CPlusPlusBase() {}

public:
  virtual ~DAT_CPlusPlusBase() {}

  // BEGIN PUBLIC INTERFACE

  // Some general settings can be assigned here (if you ovierride it)

  virtual void getGeneralInfo(DAT_GeneralInfo *, const OP_Inputs *,
                              void *reserved1) {}

  // Add geometry data such as points, normals, colors, and triangles
  // or particles and etc. obtained from your desired algorithm or external
  // files. If the "directToGPU" flag is set to false, this function is being
  // called instead of executeVBO(). See the OP_Inputs class definition for more
  // details on it's contents
  virtual void execute(DAT_Output *, const OP_Inputs *, void *reserved1) = 0;

  // Override these methods if you want to output values to the Info CHOP/DAT
  // returning 0 means you dont plan to output any Info CHOP channels
  virtual int32_t getNumInfoCHOPChans(void *reserved1) { return 0; }

  // Specify the name and value for CHOP 'index',
  // by assigning something to 'name' and 'value' members of the
  // OP_InfoCHOPChan class pointer that is passed (it points
  // to a valid instance of the class already.
  // the 'name' pointer will initially point to nullptr
  // you must allocate memory or assign a constant string
  // to it.
  virtual void getInfoCHOPChan(int32_t index, OP_InfoCHOPChan *chan,
                               void *reserved1) {}

  // Return false if you arn't returning data for an Info DAT
  // Return true if you are.
  // Set the members of the CHOP_InfoDATSize class to specify
  // the dimensions of the Info DAT
  virtual bool getInfoDATSize(OP_InfoDATSize *infoSize, void *reserved1) {
    return false;
  }

  // You are asked to assign values to the Info DAT 1 row or column at a time
  // The 'byColumn' variable in 'getInfoDATSize' is how you specify
  // if it is by column or by row.
  // 'index' is the row/column index
  // 'nEntries' is the number of entries in the row/column
  virtual void getInfoDATEntries(int32_t index, int32_t nEntries,
                                 OP_InfoDATEntries *entries, void *reserved1) {}

  // You can use this function to put the node into a warning state
  // with the returned string as the message.
  virtual void getWarningString(OP_String *warning, void *reserved1) {}

  // You can use this function to put the node into a error state
  // with the returned string as the message.
  virtual void getErrorString(OP_String *error, void *reserved1) {}

  // Use this function to return some text that will show up in the
  // info popup (when you middle click on a node)
  virtual void getInfoPopupString(OP_String *info, void *reserved1) {}

  // Override these methods if you want to define specfic parameters
  virtual void setupParameters(OP_ParameterManager *manager, void *reserved1) {}

  // This is called whenever a pulse parameter is pressed
  virtual void pulsePressed(const char *name, void *reserved1) {}

  // This is called whenever a dynamic menu type custom parameter needs to have
  // it's content's updated. It may happen often, so this could should be
  // efficient.
  virtual void buildDynamicMenu(const OP_Inputs *inputs,
                                OP_BuildDynamicMenuInfo *info,
                                void *reserved1) {}

  // END PUBLIC INTERFACE

private:
  // Reserved for future features
  virtual int32_t reservedFunc6() { return 0; }
  virtual int32_t reservedFunc7() { return 0; }
  virtual int32_t reservedFunc8() { return 0; }
  virtual int32_t reservedFunc9() { return 0; }
  virtual int32_t reservedFunc10() { return 0; }
  virtual int32_t reservedFunc11() { return 0; }
  virtual int32_t reservedFunc12() { return 0; }
  virtual int32_t reservedFunc13() { return 0; }
  virtual int32_t reservedFunc14() { return 0; }
  virtual int32_t reservedFunc15() { return 0; }
  virtual int32_t reservedFunc16() { return 0; }
  virtual int32_t reservedFunc17() { return 0; }
  virtual int32_t reservedFunc18() { return 0; }
  virtual int32_t reservedFunc19() { return 0; }
  virtual int32_t reservedFunc20() { return 0; }

  int32_t reserved[400];
};

#pragma pack(pop)

static_assert(offsetof(DAT_PluginInfo, apiVersion) == 0, "Incorrect Alignment");
static_assert(offsetof(DAT_PluginInfo, customOPInfo) == 408,
              "Incorrect Alignment");
static_assert(sizeof(DAT_PluginInfo) == 944, "Incorrect Size");

static_assert(offsetof(DAT_GeneralInfo, cookEveryFrame) == 0,
              "Incorrect Alignment");
static_assert(offsetof(DAT_GeneralInfo, cookEveryFrameIfAsked) == 1,
              "Incorrect Alignment");
static_assert(sizeof(DAT_GeneralInfo) == 84, "Incorrect Size");

}; // namespace TD

#endif
