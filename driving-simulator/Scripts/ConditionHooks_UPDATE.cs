// =============================================================================
// CONDITION-AWARE HOOKS — paste into QTQuestionnaireManager.cs
// =============================================================================


// =============================================================================
// 1. REPLACE EndOfQt() with this:
// =============================================================================

        private void EndOfQt()
        {
            // Check if study is already finished (final questionnaire completed)
            var loopForQt = GameObject.FindGameObjectWithTag("LoopForQT")
                .GetComponent<LoopForQT>();
            if (loopForQt.studyFinished)
            {
                Debug.Log("[EndOfQt] Study already finished — skipping.");
                return;
            }

            var bomanager = GameObject.FindGameObjectWithTag("BOforUnityManager")
                ?.GetComponent<BoForUnityManager>();
            var carController = GameObject.Find("AI_Car").GetComponent<CarAIController>();

            // --- Log current condition ---
            string currentCondition = "combination";
            if (bomanager != null)
            {
                currentCondition = StudyConditionManager.GetCondition(bomanager.conditionId);
                Debug.Log($"[EndOfQt] Condition={currentCondition} " +
                          $"({StudyConditionManager.GetConditionDisplayName(currentCondition)}) " +
                          $"ConditionId={bomanager.conditionId} Group={bomanager.groupId}");
            }

            // --- EMOTION: apply emotion to objectives (condition-aware) ---
            var emotionBridge = FindObjectOfType<EmotionBOBridge>();
            if (emotionBridge != null)
            {
                emotionBridge.ApplyEmotionToObjectives();
            }
            else
            {
                Debug.LogWarning("[EndOfQt] EmotionBOBridge not found.");
            }
            // --- END EMOTION ---

            if (bomanager != null && loopForQt.CURRENT_IT <= carController.totalIterationsFixed)
            {
                Debug.Log("TotalIterations" + carController.totalIterationsFixed);
                Debug.Log("CurrentIteration : " + loopForQt.CURRENT_IT);
                Debug.Log("QTPaused: " + loopForQt.pauseQt);

                if (loopForQt.CURRENT_IT == carController.totalIterationsFixed)
                {
                    loopForQt.timer += loopForQt.timerExtension;
                    bomanager.OptimizationStart();

                    Debug.Log("FinalDesign via QT with iteration: " + loopForQt.CURRENT_IT);
                    bomanager.SelectAndApplyFinalDesign();
                    carController.updatePopUp(true, "The System decided on your final design. You will experience this design for the remainder of the environment.");
                }
                else
                {
                    bomanager.OptimizationStart();
                    Debug.Log("BOStarted via QT");
                }
            }

            if (loopForQt.CURRENT_IT > carController.totalIterationsFixed || bomanager == null)
            {
                Debug.Log("Totalits " + carController.totalIterations);

                if (loopForQt.CURRENT_IT == carController.totalIterations)
                {
                    loopForQt.timer += loopForQt.timerExtension;
                }

                if (loopForQt.CURRENT_IT == carController.totalIterations + 1)
                {
                    loopForQt.timer -= loopForQt.timerExtension;
                    loopForQt.AfterAreaQt();
                }
                else
                {
                    Debug.Log("QTStarted");
                    Debug.Log("CurrentIteration : " + loopForQt.CURRENT_IT);
                    Debug.Log("TotalIts : " + carController.totalIterationsFixed);
                    loopForQt.StartQuestionnaireLoop();
                }
            }
        }


// =============================================================================
// 2. REPLACE WriteResults() with this:
// =============================================================================

        private void WriteResults()
        {
            StreamWriter writer = new StreamWriter(userPath, true);
            var line = currentResponseId + ",";
            if (customUserId)
            {
                line += "\"" + userId + "\",";
                if (runsPerUser == 1)
                    userId = "";
            }
            if (runsPerUser > 1)
            {
                line += currentRun + ",";
            }
            if (generateStartTimestamp)
            {
                line += startedTimestamp + ",";
            }
            if (generateFinishTimestamp)
            {
                line += finishedTimestamp + ",";
            }

            //-----------
            var boManager = false;
            var boForUnity = FindObjectOfType<BoForUnityManager>();
            if (boForUnity != null)
            {
                boManager = true;
            }
            var boCounter = 0;

            // Track questionnaire values for unified CSV
            var questionnaireValues = new Dictionary<string, List<float>>();

            // Clear previous iteration's objective values before adding new ones
            if (boManager)
            {
                foreach (var ob in boForUnity.objectives)
                {
                    ob.value.values.Clear();
                }
            }

            // Determine if questionnaire values should go to BO
            bool useQuestionnaireForBO = true;
            if (boManager)
            {
                useQuestionnaireForBO = StudyConditionManager.ShouldUseQuestionnaireForBO(boForUnity.conditionId);

                if (!useQuestionnaireForBO)
                {
                    string condition = StudyConditionManager.GetCondition(boForUnity.conditionId);
                    Debug.Log($"[WriteResults] Condition={condition} — questionnaire values logged but NOT sent to BO.");
                }
            }
            //-----------
            
            var currVal = "";
            
            foreach (var page in questionPages)
            {
                page.SetActive(true);
                
                foreach (var question in page.GetComponent<QTQuestionPageManager>().questionItems)
                {
                    switch (question.tag)
                    {
                        case "QTLinearScale":
                            var toggleGroup = question.transform.GetChild(1).GetComponent<ToggleGroup>();
                            currVal = toggleGroup.ActiveToggles().FirstOrDefault().gameObject.name.Split('_')[0];
                            line += currVal + ",";
                            break;
                        case "QTCheckboxes":
                            var checkboxesResults = "";
                            for (var c = 0; c < question.transform.GetChild(1).childCount; c++)
                            {
                                var checkbox = question.transform.GetChild(1).GetChild(c).GetComponent<Toggle>();
                                if (checkbox.isOn)
                                {
                                    if (checkbox.CompareTag("QTOptionOther"))
                                    {
                                        checkboxesResults += "" + checkbox.transform.GetChild(1).GetComponent<TMP_InputField>().text + ";";
                                    }
                                    else
                                    {
                                        checkboxesResults += checkbox.name.Split('_')[0] + ";";
                                    }
                                }
                            }
                            currVal = checkboxesResults.TrimEnd(';');
                            line += currVal + ",";
                            break;
                        case "QTSlider":
                            currVal = question.transform.GetChild(1).GetComponent<UnityEngine.UI.Slider>().value + "";
                            line += currVal + ",";
                            break;
                        case "QTMultipleChoice":
                            var toggleGroupMc = question.transform.GetChild(1).GetComponent<ToggleGroup>();
                            var toggledOption = toggleGroupMc.ActiveToggles().FirstOrDefault();
                            if (toggledOption.CompareTag("QTOptionOther"))
                            {
                                currVal = toggledOption.transform.GetChild(1).GetComponent<TMP_InputField>().text;
                                line += "\"" + currVal + "\",";
                            }
                            else
                            {
                                currVal = toggledOption.gameObject.name.Split('_')[0];
                                line += currVal + ",";
                            }
                            break;
                        case "QTTextInput":
                            currVal = question.transform.GetChild(1).GetComponent<TMP_InputField>().text;
                            line += "\"" + currVal + "\",";
                            break;
                        case "QTDropdown":
                            var tmp = question.transform.GetChild(1).GetComponent<TMP_Dropdown>().value;
                            currVal = question.transform.GetChild(1).GetComponent<TMP_Dropdown>().options[tmp].text;
                            line += "\"" + currVal + "\",";
                            break;
                        case "QTCheckboxesGrid":
                            var checkboxGrid = question.transform.GetChild(1);
                            var optionCounter = 0;
                            var columnCount = question.GetComponent<QTCheckboxesGrid>().columnTexts.Count;
                            for (var i = 0; i < checkboxGrid.childCount; i++)
                            {
                                var currChild = checkboxGrid.GetChild(i);
                                if (currChild.CompareTag("QTGridOption"))
                                {
                                    if (currChild.GetComponent<Toggle>().isOn)
                                    {
                                        currVal = currChild.name.Split('_')[1];
                                        line += "" + currVal + ";";
                                    }
                                    optionCounter++;
                                }
                                if (optionCounter == columnCount)
                                {
                                    optionCounter = 0;
                                    line = line.TrimEnd(';') + ",";
                                }
                            }
                            break;
                        case "QTMultipleChoiceGrid":
                            var grid = question.transform.GetChild(1);
                            for (var i = 0; i < grid.childCount; i++)
                            {
                                var currChild = grid.GetChild(i);
                                if (currChild.CompareTag("QTGridRowHeader"))
                                {
                                    var toggleGroupRow = currChild.GetComponent<ToggleGroup>();
                                    currVal = toggleGroupRow.ActiveToggles().FirstOrDefault().gameObject.name.Split('_')[1];
                                    line += "" + currVal + ",";
                                }
                            }
                            break;
                    }
                    
                    // Add the current question item value as objective function value
                    if (boManager)
                    {
                        // Track the actual questionnaire value for unified CSV
                        string headerName = resultsHeaderItems[boCounter];
                        float parsedVal = 0f;
                        float.TryParse(currVal, CultureInfo.InvariantCulture, out parsedVal);

                        // Find which objective this belongs to
                        foreach (var ob in boForUnity.objectives)
                        {
                            if (headerName.StartsWith(ob.key) || headerName == ob.key)
                            {
                                if (!questionnaireValues.ContainsKey(ob.key))
                                    questionnaireValues[ob.key] = new List<float>();
                                questionnaireValues[ob.key].Add(parsedVal);
                                break;
                            }
                        }

                        if (useQuestionnaireForBO)
                        {
                            // value_only or combination: questionnaire values go to BO
                            boForUnity.optimizer.AddObjectiveValue(
                                resultsHeaderItems[boCounter],
                                float.Parse(currVal, CultureInfo.InvariantCulture));
                        }
                        else
                        {
                            // llm_only: log in CSV but send 0 to BO
                            boForUnity.optimizer.AddObjectiveValue(
                                resultsHeaderItems[boCounter], 0f);
                            Debug.Log($"[WriteResults] LLM-only — '{resultsHeaderItems[boCounter]}'={currVal} NOT sent to BO.");
                        }
                    }
                    boCounter++;
                }
            }

            foreach (var aci in additionalCsvItems)
            {
                if (aci.itemValue.isAssigned)
                {
                    currVal = aci.itemValue.Get() + "";
                    line += "\"" + currVal + "\",";
                    
                    if(boCounter < resultsHeaderItems.Count()){
                        Debug.LogWarning("You have more additional csv items assigned than question items in the questionnaire. Please add the corresponding header names to the resultsHeaderItems list in the inspector of the QTQuestionnaireManager or reduce the number of additional csv items.");
                        if (boManager)
                        {
                            if (useQuestionnaireForBO)
                            {
                                boForUnity.optimizer.AddObjectiveValue(
                                    resultsHeaderItems[boCounter],
                                    float.Parse(currVal, CultureInfo.InvariantCulture));
                            }
                            else
                            {
                                boForUnity.optimizer.AddObjectiveValue(
                                    resultsHeaderItems[boCounter], 0f);
                                Debug.Log($"[WriteResults] LLM-only — '{resultsHeaderItems[boCounter]}'={currVal} NOT sent to BO.");
                            }
                        }
                        boCounter++;
                    }
                    Debug.Log("CSV currVal: " + currVal + "BOCounter: " + boCounter + "ResultHeaderItemsMaxSize " + resultsHeaderItems.Count());
                }
                else
                {
                    line += "\"NULL\",";
                }
            }

            // Store questionnaire values for unified CSV logging
            var emotionBridge = FindObjectOfType<EmotionBOBridge>();
            if (emotionBridge != null && questionnaireValues.Count > 0)
            {
                emotionBridge.StoreQuestionnaireValues(questionnaireValues);
            }

            try
            {
                writer.WriteLine(line.TrimEnd(','));
                writer.Close();
                Debug.Log("Write results for: " + resultsFileName + (newFileEachStart ? "_" + currentResponseId : "") + 
                          (runsPerUser > 1 ? " run: " + currentRun : "") + ".\n@ " + userPath);
                if (currentRun == runsPerUser)
                {
                    qtMetaData.currentUserRun = 0;
                    currentRun = 0;
                    userId = "";
                }
                SaveMetaData(currentResponseId, currentRun);
            }
            catch (Exception)
            {
                Debug.Log("Write results failed!");
            }
        }
