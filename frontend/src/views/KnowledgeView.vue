<template>
  <section class="page-shell knowledge-chat-page">
    <aside class="chat-session-panel panel">
      <header class="chat-panel-head">
        <div>
          <p class="eyebrow">知识对话</p>
          <h2>知识库对话</h2>
        </div>
        <button class="button-secondary" :disabled="store.loading" @click="store.createNewSession">
          <Plus :size="16" />
          新建对话
        </button>
      </header>

      <div class="panel-body panel-scroll">
        <ul class="chat-session-list">
          <li v-for="session in store.sessions" :key="session.id" class="chat-session-row">
            <button
              class="chat-session-button"
              :class="{ 'chat-session-active': session.id === store.currentSessionId }"
              @click="store.openSession(session.id)"
              :title="sessionSummary(session)"
            >
              <span class="chat-session-title">{{ sessionSummary(session) }}</span>
            </button>
            <button
              class="chat-session-delete"
              type="button"
              aria-label="删除对话"
              title="删除对话"
              :disabled="store.loading"
              @click.stop="store.removeSession(session.id)"
            >
              <Trash2 :size="15" aria-hidden="true" />
            </button>
          </li>
        </ul>

        <section class="workbench-file-panel" aria-label="Workbench 文件区">
          <header class="workbench-file-head">
            <div>
              <p class="eyebrow">Workbench</p>
              <h3>文件上下文</h3>
            </div>
            <button
              class="icon-button"
              type="button"
              title="刷新文件上下文"
              aria-label="刷新文件上下文"
              :disabled="store.isWorkbenchLoading"
              @click="store.refreshWorkbenchFileContext"
            >
              <RefreshCcw :size="15" />
            </button>
          </header>

          <div v-if="store.isWorkbenchLoading && !workbenchDocuments.length && !workbenchSessionFiles.length" class="empty-state">
            正在加载文件上下文...
          </div>

          <section class="workbench-file-section workbench-library-section" aria-label="论文库文件">
            <div class="workbench-section-head">
              <div>
                <h4>论文库文件</h4>
                <p>PDF 论文可勾选后用于论文问答。</p>
              </div>
              <button
                class="button-secondary workbench-upload-button"
                type="button"
                :disabled="documentStore.submittingUpload"
                @click="triggerPdfInput"
              >
                <UploadCloud :size="15" />
                {{ documentStore.submittingUpload ? "上传中..." : "上传 PDF" }}
              </button>
            </div>

            <ul v-if="workbenchDocuments.length" class="workbench-file-list">
              <li v-for="document in workbenchDocuments" :key="document.id">
                <button
                  class="workbench-file-row"
                  type="button"
                  :class="{ 'workbench-file-selected': isWorkbenchDocumentSelected(document.id) }"
                  :disabled="document.status !== 'ready'"
                  :title="document.display_name"
                  @click="toggleWorkbenchDocument(document)"
                >
                  <span class="workbench-file-check" aria-hidden="true">
                    {{ isWorkbenchDocumentSelected(document.id) ? "✓" : "" }}
                  </span>
                  <span class="workbench-file-copy">
                    <strong>{{ document.display_name }}</strong>
                    <small>{{ document.title || document.filename }}</small>
                    <span class="workbench-file-badges">
                      <span class="status-badge" :data-status="document.status">
                        {{ formatDocumentStatus(document.status) }}
                      </span>
                      <span v-if="isWorkbenchDocumentUsed(document.id)" class="workbench-mini-badge">已用</span>
                      <span v-if="isWorkbenchDocumentRecent(document.id)" class="workbench-mini-badge">最近</span>
                    </span>
                  </span>
                </button>
              </li>
            </ul>
            <p v-else-if="!store.isWorkbenchLoading" class="empty-state">
              暂无库内论文。
            </p>
          </section>

          <section class="workbench-file-section workbench-session-section" aria-label="会话文件">
            <div class="workbench-section-head">
              <div>
                <h4>会话文件</h4>
                <p>仅展示，暂未接入 Chat。支持 txt、md、docx，单个文件最大 5MB。</p>
              </div>
              <button
                class="button-secondary workbench-upload-button"
                type="button"
                :disabled="store.isUploadingSessionFile"
                @click="triggerSessionFileInput"
              >
                <UploadCloud :size="15" />
                {{ store.isUploadingSessionFile ? "上传中..." : "上传文件" }}
              </button>
            </div>

            <p v-if="store.sessionFileUploadError" class="error-text">
              {{ store.sessionFileUploadError }}
            </p>

            <ul v-if="workbenchSessionFiles.length" class="workbench-session-file-list">
              <li v-for="file in workbenchSessionFiles" :key="file.id">
                <button
                  class="workbench-session-file-card"
                  type="button"
                  :class="{
                    'workbench-session-file-selected': isWorkbenchSessionFileSelected(file),
                    'workbench-session-file-disabled': !isSessionFileSelectable(file)
                  }"
                  :disabled="!isSessionFileSelectable(file)"
                  :title="file.display_name || file.filename"
                  @click="toggleWorkbenchSessionFile(file)"
                >
                  <div class="session-file-main">
                    <span class="session-file-icon" aria-hidden="true">{{ formatFileKind(file.kind) }}</span>
                    <span class="workbench-file-copy">
                      <strong>{{ file.display_name || file.filename }}</strong>
                      <small>
                        {{ formatFileMeta(file) }}
                      </small>
                    </span>
                  </div>
                  <span class="workbench-file-badges">
                    <span class="status-badge" :data-status="file.status">
                      {{ formatSessionFileStatus(file.status) }}
                    </span>
                    <span class="status-badge" :data-status="file.text_extract_status">
                      文本{{ formatTextExtractionStatus(file.text_extract_status) }}
                    </span>
                    <span v-if="isWorkbenchSessionFileSelected(file)" class="workbench-mini-badge">已选</span>
                    <span v-if="isWorkbenchSessionFileUsed(file)" class="workbench-mini-badge">已用</span>
                    <span v-if="!isSessionFileSelectable(file)" class="workbench-mini-badge workbench-mini-badge-muted">不可用</span>
                  </span>
                  <p v-if="file.status === 'ready' && file.preview_text" class="session-file-preview">
                    {{ file.preview_text }}
                  </p>
                  <p v-if="sessionFileReason(file)" class="session-file-reason">
                    {{ sessionFileReason(file) }}
                  </p>
                  <p class="session-file-note">
                    {{ isSessionFileSelectable(file) ? "用于本轮 Chat" : "普通文件只读上下文" }}
                  </p>
                </button>
              </li>
            </ul>
            <p v-else-if="!store.isWorkbenchLoading" class="empty-state">
              暂无会话文件。
            </p>
          </section>
        </section>
      </div>
    </aside>

    <article class="panel chat-stage">
      <header class="chat-stage-head">
        <div>
          <p class="eyebrow">PaperDesk 助手</p>
          <h2>{{ store.currentSession?.title || "新对话" }}</h2>
          <p>可直接提问，也可以附加图片或论文材料，再生成问答、总结、综述与对比结果。</p>
        </div>
      </header>

      <div ref="messagePanelRef" class="panel-body panel-scroll chat-message-panel">
        <div v-if="store.loading" class="empty-state">正在加载会话…</div>

        <div v-else-if="!store.messages.length" class="chat-empty-state">
          <div class="chat-empty-copy">
            <h3>开始一轮新的知识对话</h3>
            <p>你可以直接提问，也可以先附加图片、刚上传的论文 PDF，或从论文库里选择已入库文档。</p>
          </div>
        </div>

        <ul v-else class="chat-message-list">
          <li
            v-for="message in store.messages"
            :key="message.id"
            class="chat-message-item"
            :data-role="message.role"
            :class="{ 'chat-message-trace-active': message.role === 'assistant' && message.id === store.selectedTraceMessageId }"
            @click="selectTraceMessage(message)"
          >
            <div class="chat-message-bubble">
              <div class="chat-message-meta">
                <strong>{{ formatRole(message.role) }}</strong>
                <span>{{ formatTime(message.created_at) }}</span>
              </div>

              <div v-if="message.attachments.length" class="chat-attachment-grid">
                <article
                  v-for="attachment in message.attachments"
                  :key="attachment.id"
                  class="chat-attachment-card"
                >
                  <img
                    v-if="attachment.kind === 'image' && attachment.data_url"
                    :src="attachment.data_url"
                    :alt="attachment.display_name"
                    class="chat-attachment-image"
                  />
                  <div>
                    <strong>{{ attachment.display_name }}</strong>
                    <p class="card-meta">
                      {{ formatAttachmentKind(attachment.kind) }}
                      <span v-if="attachment.status"> · {{ attachment.status }}</span>
                    </p>
                  </div>
                </article>
              </div>

              <MarkdownPreview
                v-if="message.role === 'assistant' && message.content"
                :markdown="message.content"
              />
              <div v-if="message.role === 'assistant' && isMessageProcessing(message)" class="chat-processing-indicator">
                <span>正在处理</span>
                <i></i>
                <i></i>
                <i></i>
              </div>
              <p v-if="message.role !== 'assistant'" class="chat-message-text">{{ message.content }}</p>

              <p v-if="message.warning" class="hint-text">{{ message.warning }}</p>

              <div class="chat-message-actions">
                <button
                  class="button-secondary message-action-button"
                  :disabled="!message.content.trim()"
                  @click="copyMessage(message)"
                >
                  <Copy :size="15" />
                  {{ copiedMessageId === message.id ? "已复制" : "复制" }}
                </button>
                <button
                  v-if="message.role === 'assistant'"
                  class="button-secondary message-action-button"
                  :disabled="isMessageProcessing(message) || !message.content.trim() || Boolean(message.saved_report_id) || store.savingReportMessageId === message.id"
                  @click="saveMessageReport(message)"
                >
                  <Save :size="15" />
                  {{
                    message.saved_report_id
                      ? "已保存"
                      : store.savingReportMessageId === message.id
                        ? "保存中..."
                        : "保存为报告"
                  }}
                </button>
                <button
                  v-if="message.role === 'assistant'"
                  class="button-secondary message-action-button"
                  :disabled="isMessageProcessing(message)"
                  @click.stop="showMessageTrace(message)"
                >
                  <FileText :size="15" />
                  执行摘要
                </button>
              </div>

            </div>
          </li>
        </ul>
      </div>

      <div class="chat-composer">
        <div v-if="store.retrievalNotice" class="hint-text">{{ store.retrievalNotice }}</div>
        <p v-if="store.error" class="error-text">{{ store.error }}</p>
        <p v-if="documentStore.error" class="error-text">{{ documentStore.error }}</p>

        <div v-if="visibleDraftAttachments.length" class="draft-strip">
          <div class="memory-chip-list">
            <button
              v-for="attachment in visibleDraftAttachments"
              :key="attachment.id"
              class="draft-chip"
              @click="store.removeDraftAttachment(attachment.id)"
            >
              <span>{{ attachment.display_name }}</span>
              <small>{{ formatAttachmentKind(attachment.kind) }}</small>
            </button>
          </div>
        </div>

        <div v-if="showDocumentPicker" class="document-picker-sheet">
          <div class="sheet-head">
            <strong>从论文库选择</strong>
          <button class="button-secondary" @click="showDocumentPicker = false">收起</button>
          </div>
          <div class="document-pick-grid">
            <button
              v-for="document in readyDocuments"
              :key="document.id"
              class="pick-card pick-card-button"
              :class="{ 'task-card-active': store.selectedDocumentIds.includes(document.id) }"
              @click="store.toggleLibraryDocument(document)"
            >
              <strong>{{ document.display_name }}</strong>
              <p class="card-meta">{{ document.title || "未提取标题" }}</p>
              <p class="card-meta">{{ document.page_count || 0 }} 页 · {{ formatDocumentStatus(document.status) }}</p>
            </button>
            <p v-if="!readyDocuments.length" class="empty-state">
              暂无可选择的已入库论文，请先到本地论文库上传并等待处理完成。
            </p>
          </div>
        </div>

        <div v-if="showAttachMenu" class="attach-popover">
          <button class="button-secondary" @click="triggerImageInput">上传图片</button>
          <button class="button-secondary" @click="triggerPdfInput">上传本地 PDF</button>
          <button class="button-secondary" @click="toggleDocumentPicker">选择库内论文</button>
        </div>

        <div v-if="showSlashMenu" class="slash-command-menu" role="listbox" aria-label="Slash commands">
          <button
            v-for="(command, index) in filteredSlashCommands"
            :key="command.id"
            class="slash-command-row"
            :class="{ 'slash-command-row-active': index === slashCommandIndex }"
            type="button"
            role="option"
            :aria-selected="index === slashCommandIndex"
            @mousedown.prevent="selectSlashCommand(command)"
          >
            <span class="slash-command-label">{{ command.label }}</span>
            <span class="slash-command-copy">
              <strong>{{ command.description }}</strong>
              <small>{{ command.group }}</small>
            </span>
          </button>
        </div>

        <div class="composer-shell">
          <div
            class="composer-resize-handle"
            role="separator"
            aria-label="调整输入框高度"
            aria-orientation="horizontal"
            @pointerdown="startComposerResize"
          ></div>
          <div v-if="selectedLibraryDraftAttachments.length" class="selected-document-strip" aria-label="已选论文">
            <article
              v-for="attachment in selectedLibraryDraftAttachments"
              :key="attachment.id"
              class="selected-document-chip"
              :title="attachment.display_name"
            >
              <span class="selected-document-icon" aria-hidden="true">
                <FileText :size="18" />
              </span>
              <span class="selected-document-copy">
                <strong>{{ attachment.display_name }}</strong>
                <small>
                  {{ formatAttachmentKind(attachment.kind) }}
                  <span v-if="attachment.status"> · {{ formatDocumentStatus(attachment.status) }}</span>
                </small>
              </span>
              <button
                class="selected-document-remove"
                type="button"
                :aria-label="`移除 ${attachment.display_name}`"
                @click="store.removeDraftAttachment(attachment.id)"
              >
                <X :size="14" />
              </button>
            </article>
          </div>
          <p v-if="mixedSelectionHint" class="mixed-selection-hint">{{ mixedSelectionHint }}</p>
          <div v-if="selectedSessionFileChips.length" class="selected-document-strip selected-file-strip" aria-label="已选普通文件">
            <article
              v-for="file in selectedSessionFileChips"
              :key="getSessionFileId(file)"
              class="selected-document-chip selected-file-chip"
              :title="file.display_name || file.filename"
            >
              <span class="selected-document-icon selected-file-icon" aria-hidden="true">
                {{ formatFileKind(file.kind) }}
              </span>
              <span class="selected-document-copy">
                <strong>{{ file.display_name || file.filename }}</strong>
                <small>{{ formatFileKind(file.kind) }} · {{ formatBytes(file.size_bytes) }}</small>
              </span>
              <button
                class="selected-document-remove"
                type="button"
                :aria-label="`移除 ${file.display_name || file.filename}`"
                @click="store.toggleSessionFile(getSessionFileId(file))"
              >
                <X :size="14" />
              </button>
            </article>
          </div>
          <button class="composer-plus" aria-label="添加附件" @click="showAttachMenu = !showAttachMenu">
            <Paperclip :size="19" />
          </button>
          <div v-if="store.activeSlashCommand" class="slash-command-token">
            <span>{{ store.activeSlashCommand.label }}</span>
            <button type="button" aria-label="Clear slash command" @click="store.clearSlashCommand">
              <X :size="13" />
            </button>
          </div>
          <p v-if="slashCommandHint" class="slash-command-hint">{{ slashCommandHint }}</p>
          <div v-if="store.activeSlashCommand?.id === 'help'" class="slash-help-card">
            <strong>Slash commands</strong>
            <span v-for="command in slashCommands" :key="command.id">
              {{ command.label }} - {{ command.description }}
            </span>
          </div>
          <textarea
            v-model="store.composerText"
            class="chat-textarea"
            :style="{ height: `${composerHeight}px` }"
            placeholder="输入问题，或附加 PDF / 库内论文后发送。"
            @keydown="handleKeydown"
            @input="handleComposerInput"
          />
          <button
            class="button-primary composer-send"
            :disabled="!store.sending && !canSendMessage"
            @click="handleComposerAction"
          >
            <StopCircle v-if="store.sending" :size="16" />
            <SendHorizontal v-else :size="16" />
            {{ store.sending ? (store.stopping ? "停止中..." : "停止") : "发送" }}
          </button>
        </div>
      </div>
    </article>

    <aside class="capability-panel panel" aria-label="Workbench capabilities">
      <header class="capability-panel-head">
        <div>
          <p class="eyebrow">Workbench</p>
          <h3>{{ rightPanelTab === "capabilities" ? "Agent capabilities" : "Trace / Artifacts" }}</h3>
        </div>
        <button
          v-if="rightPanelTab === 'capabilities'"
          class="icon-button"
          type="button"
          title="Refresh capabilities"
          aria-label="Refresh capabilities"
          :disabled="store.isCapabilitiesLoading"
          @click="store.loadWorkbenchCapabilities"
        >
          <RefreshCcw :size="15" />
        </button>
        <button
          v-else
          class="icon-button"
          type="button"
          title="Refresh trace"
          aria-label="Refresh trace"
          :disabled="store.isTraceLoading || !store.selectedTraceMessageId"
          @click="store.refreshSelectedMessageTrace"
        >
          <RefreshCcw :size="15" />
        </button>
      </header>

      <div class="workbench-side-tabs" role="tablist" aria-label="Workbench side panel">
        <button
          type="button"
          :class="{ 'workbench-side-tab-active': rightPanelTab === 'capabilities' }"
          @click="rightPanelTab = 'capabilities'"
        >
          Capabilities
        </button>
        <button
          type="button"
          :class="{ 'workbench-side-tab-active': rightPanelTab === 'trace' }"
          @click="activateTraceTab"
        >
          Trace / Artifacts
        </button>
      </div>

      <div class="panel-body panel-scroll capability-panel-body">
        <div v-if="rightPanelTab === 'capabilities' && store.isCapabilitiesLoading && !store.workbenchCapabilities" class="empty-state">
          Loading capabilities...
        </div>
        <template v-else-if="rightPanelTab === 'capabilities'">
          <section class="capability-section">
            <h4>Stable capabilities</h4>
            <div class="capability-list">
              <button
                v-for="capability in stableCapabilityItems"
                :key="capability.id"
                class="capability-item"
                type="button"
                :disabled="!canApplyCapabilitySlashCommand(capability)"
                @click="applyCapabilitySlashCommand(capability)"
              >
                <span class="capability-title-row">
                  <strong>{{ capability.name }}</strong>
                  <small v-if="capability.slash_command">/{{ capability.slash_command }}</small>
                </span>
                <span>{{ capability.description }}</span>
                <span class="capability-badges">
                  <i>{{ formatCapabilityIoType(capability.io_type) }}</i>
                  <i>{{ formatOperationLevel(capability.operation_level) }}</i>
                  <i v-if="capability.requires_confirmation">Requires confirmation</i>
                  <i v-if="capability.destructive" class="capability-risk-badge">High risk</i>
                  <i v-if="!capability.current_available">Unavailable</i>
                </span>
              </button>
            </div>
          </section>

          <section class="capability-section">
            <h4>Needs confirmation</h4>
            <div class="capability-list">
              <article
                v-for="capability in confirmationCapabilityItems"
                :key="capability.id"
                class="capability-item capability-item-static"
              >
                <span class="capability-title-row">
                  <strong>{{ capability.name }}</strong>
                  <small>No direct execution</small>
                </span>
                <span>{{ capability.description }}</span>
                <span class="capability-badges">
                  <i>{{ formatCapabilityIoType(capability.io_type) }}</i>
                  <i>Requires confirmation</i>
                  <i v-if="capability.destructive" class="capability-risk-badge">Destructive</i>
                </span>
              </article>
              <p v-if="!confirmationCapabilityItems.length" class="empty-state">
                No stable capability currently requires confirmation.
              </p>
            </div>
          </section>

          <section class="capability-section">
            <h4>Slash command</h4>
            <div class="capability-command-grid">
              <button
                v-for="command in capabilitySlashCommands"
                :key="command.id"
                class="capability-command"
                type="button"
                @click="applySlashCommandById(command.id)"
              >
                <strong>{{ command.label }}</strong>
                <span>{{ command.description }}</span>
              </button>
            </div>
          </section>

          <section class="capability-section">
            <h4>Experimental area</h4>
            <div class="capability-list">
              <article
                v-for="capability in experimentalCapabilityItems"
                :key="capability.id"
                class="capability-item capability-item-static capability-item-experimental"
              >
                <span class="capability-title-row">
                  <strong>{{ capability.name }}</strong>
                  <small>Future stage</small>
                </span>
                <span>{{ capability.description }}</span>
                <span class="capability-badges">
                  <i>Experimental</i>
                  <i>Unavailable</i>
                </span>
              </article>
            </div>
          </section>
        </template>
        <template v-else>
          <div v-if="store.isTraceLoading" class="empty-state">
            Loading trace summary...
          </div>
          <div v-else-if="store.traceError" class="trace-empty-state">
            <strong>Trace unavailable</strong>
            <span>{{ store.traceError }}</span>
          </div>
          <div v-else-if="!latestAssistantMessage" class="trace-empty-state">
            <strong>No assistant message yet</strong>
            <span>Send a message to see its execution summary here.</span>
          </div>
          <div v-else-if="!traceSummary" class="trace-empty-state">
            <strong>No trace selected</strong>
            <span>Select an assistant message to inspect its compact summary.</span>
          </div>
          <div v-else class="trace-summary-panel">
            <section class="trace-section">
              <div class="trace-message-row">
                <strong>{{ selectedTraceMessageLabel }}</strong>
                <span>{{ traceSummary.trace_id ? shortId(traceSummary.trace_id) : "No trace id" }}</span>
              </div>
              <div class="trace-status-grid">
                <span>
                  <small>Route</small>
                  <strong>{{ formatTraceValue(traceSummary.route) }}</strong>
                </span>
                <span>
                  <small>Action</small>
                  <strong>{{ formatTraceValue(traceSummary.action_status) }}</strong>
                </span>
                <span>
                  <small>Retrieval</small>
                  <strong>{{ formatTraceValue(traceSummary.retrieval_status) }}</strong>
                </span>
                <span>
                  <small>Confirm</small>
                  <strong>{{ formatConfirmationStatus(traceSummary.confirmation_status) }}</strong>
                </span>
                <span>
                  <small>Risk</small>
                  <strong>{{ formatRiskLevel(traceSummary.risk_level) }}</strong>
                </span>
                <span>
                  <small>Evidence</small>
                  <strong>{{ traceSummary.evidence_count }}</strong>
                </span>
              </div>
            </section>

            <section class="trace-section">
              <h4>自动使用的 Skill</h4>
              <div v-if="traceUsedSkills.length" class="trace-skill-list">
                <article v-for="skill in traceUsedSkills" :key="skill.skill_id" class="trace-skill-card">
                  <div class="trace-skill-head">
                    <strong>{{ skill.name }}</strong>
                    <span v-if="skill.is_primary">主要 Skill</span>
                  </div>
                  <div class="trace-skill-meta">
                    <small>置信度 {{ formatConfidence(skill.confidence) }}</small>
                    <small v-for="trigger in skill.triggered_by" :key="`${skill.skill_id}-${trigger}`">
                      {{ trigger }}
                    </small>
                  </div>
                  <p v-if="skill.trigger_reason">
                    <span>触发原因</span>
                    {{ skill.trigger_reason }}
                  </p>
                </article>
              </div>
              <p v-else class="empty-state">本轮未识别到专用 Skill</p>
            </section>

            <section class="trace-section">
              <h4>Artifacts</h4>
              <div class="trace-artifact-card">
                <strong>
                  {{
                    traceSummary.artifact_status.report_saved
                      ? "已保存报告"
                      : traceSummary.artifact_status.can_save_report
                        ? "可保存为报告"
                        : "暂无报告产物"
                  }}
                </strong>
                <span v-if="traceSummary.saved_report_id">
                  Report {{ shortId(traceSummary.saved_report_id) }}
                </span>
                <button
                  v-if="traceSummary.artifact_status.can_save_report && selectedTraceMessage"
                  class="button-secondary trace-save-button"
                  type="button"
                  :disabled="store.savingReportMessageId === selectedTraceMessage.id"
                  @click="saveMessageReport(selectedTraceMessage)"
                >
                  <Save :size="14" />
                  {{ store.savingReportMessageId === selectedTraceMessage.id ? "保存中..." : "保存为报告" }}
                </button>
              </div>
            </section>

            <section class="trace-section">
              <h4>Used papers</h4>
              <div v-if="traceUsedDocuments.length" class="trace-document-list">
                <span v-for="document in traceUsedDocuments" :key="document.id" :title="document.title">
                  <strong>{{ document.label }}</strong>
                  <small>{{ document.title }}</small>
                </span>
              </div>
              <p v-else class="empty-state">No paper was recorded for this message.</p>
            </section>

            <section class="trace-section">
              <h4>Tools</h4>
              <div v-if="traceSummary.tool_steps.length" class="trace-tool-list">
                <article v-for="(tool, index) in traceSummary.tool_steps" :key="`${tool.tool_name}-${index}`">
                  <div class="trace-tool-head">
                    <strong>{{ tool.display_name || tool.tool_name }}</strong>
                    <span>{{ formatTraceValue(tool.status) }}</span>
                  </div>
                  <p v-if="tool.summary">{{ tool.summary }}</p>
                  <div class="trace-tool-meta">
                    <small>Evidence {{ tool.evidence_count }}</small>
                    <small>{{ formatRiskLevel(tool.risk_level) }}</small>
                  </div>
                </article>
              </div>
              <p v-else class="empty-state">No tool call was recorded.</p>
            </section>

            <section class="trace-section">
              <h4>Steps</h4>
              <ol v-if="traceSummary.compact_steps.length" class="trace-step-list">
                <li v-for="(step, index) in traceSummary.compact_steps" :key="`${step.kind}-${index}-${step.created_at}`">
                  <span class="trace-step-dot" aria-hidden="true"></span>
                  <div>
                    <strong>{{ step.label }}</strong>
                    <span>{{ formatTraceValue(step.status) }} · {{ formatTime(step.created_at) }}</span>
                    <p v-if="step.detail">{{ step.detail }}</p>
                  </div>
                </li>
              </ol>
              <p v-else class="empty-state">No compact trace steps are available for this message.</p>
            </section>
          </div>
        </template>
      </div>
    </aside>

    <input
      ref="imageInputRef"
      hidden
      type="file"
      accept="image/*"
      @change="handleImageSelected"
    />
    <input
      ref="pdfInputRef"
      hidden
      type="file"
      accept=".pdf"
      @change="handlePdfSelected"
    />
    <input
      ref="sessionFileInputRef"
      hidden
      type="file"
      accept=".txt,.md,.docx"
      @change="handleSessionFileSelected"
    />
  </section>
</template>

<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { Copy, FileText, Paperclip, Plus, RefreshCcw, Save, SendHorizontal, StopCircle, Trash2, UploadCloud, X } from "lucide-vue-next";

import MarkdownPreview from "../components/MarkdownPreview.vue";
import { useDocumentStore } from "../stores/documents";
import { SLASH_COMMANDS, useKnowledgeStore } from "../stores/knowledge";
import type {
  ChatAttachment,
  ChatMessage,
  ChatMessageRole,
  ChatSession,
  SlashCommandId,
  SlashCommandOption,
  WorkbenchCapability,
  WorkbenchFileAsset,
  WorkbenchFileItem
} from "../types/models";

const store = useKnowledgeStore();
const documentStore = useDocumentStore();

const messagePanelRef = ref<HTMLElement | null>(null);
const imageInputRef = ref<HTMLInputElement | null>(null);
const pdfInputRef = ref<HTMLInputElement | null>(null);
const sessionFileInputRef = ref<HTMLInputElement | null>(null);
const showAttachMenu = ref(false);
const showDocumentPicker = ref(false);
const copiedMessageId = ref("");
const composerHeight = ref(112);
const resizeStartY = ref(0);
const resizeStartHeight = ref(0);
const resizingComposer = ref(false);
const rightPanelTab = ref<"capabilities" | "trace">("capabilities");
const sessionFileMaxBytes = 5 * 1024 * 1024;
const allowedSessionFileExtensions = new Set(["txt", "md", "docx"]);

const readyDocuments = computed(() =>
  documentStore.documents.filter((document) => document.status === "ready")
);
const slashCommands = SLASH_COMMANDS;
const slashCommandIndex = ref(0);
const slashQuery = computed(() => {
  const match = store.composerText.match(/(?:^|\s)(\/[^\s]*)$/);
  return match ? match[1].slice(1).toLowerCase() : "";
});
const filteredSlashCommands = computed(() => {
  const query = slashQuery.value;
  if (!query) {
    return slashCommands;
  }
  return slashCommands.filter((command) => command.id.startsWith(query));
});
const showSlashMenu = computed(() =>
  store.slashCommandMenuOpen && !store.activeSlashCommand && filteredSlashCommands.value.length > 0
);
const canSendMessage = computed(() => Boolean(store.composerText.trim() || store.activeSlashCommand));
const slashCommandHint = computed(() => {
  const command = store.activeSlashCommand;
  if (!command) {
    return "";
  }
  if (command.id === "compare" && store.selectedDocumentIds.length < 2) {
    return "建议先选择至少 2 篇 ready 论文；也可以继续发送，让后端按自然语言判断。";
  }
  if (command.warning) {
    return command.warning;
  }
  return "";
});
const workbenchDocuments = computed<WorkbenchFileItem[]>(() =>
  store.workbenchFileContext?.library_documents.length
    ? store.workbenchFileContext.library_documents
    : documentStore.documents
);
const workbenchUsedDocumentIds = computed(() => new Set(store.workbenchFileContext?.used_document_ids ?? []));
const workbenchRecentDocumentIds = computed(() => new Set(store.workbenchFileContext?.recent_document_ids ?? []));
const workbenchSessionFiles = computed<WorkbenchFileAsset[]>(() => {
  const files = store.workbenchFileContext?.session_files ?? [];
  const workspaceFiles = store.workbenchFileContext?.workspace_files ?? [];
  const seen = new Set<string>();
  return [...files, ...workspaceFiles].filter((file) => {
    const id = file.file_id || file.id;
    if (seen.has(id)) {
      return false;
    }
    seen.add(id);
    return true;
  });
});
const workbenchUsedFileIds = computed(() => new Set(store.workbenchFileContext?.used_file_ids ?? []));
const selectedSessionFileChips = computed(() =>
  store.selectedFileIds
    .map((fileId) => workbenchSessionFiles.value.find((file) => getSessionFileId(file) === fileId))
    .filter((file): file is WorkbenchFileAsset => Boolean(file))
);
const mixedSelectionHint = computed(() =>
  store.selectedDocumentIds.length && store.selectedFileIds.length
    ? "当前后端建议普通文件和论文库文件分开提问。"
    : ""
);
const visibleDraftAttachments = computed(() =>
  store.draftAttachments.filter((attachment) => attachment.kind !== "library_document")
);
const selectedLibraryDraftAttachments = computed(() =>
  store.draftAttachments
    .filter((attachment) => attachment.kind === "library_document")
    .map((attachment) => {
      const document = documentStore.documents.find((item) => item.id === attachment.document_id);
      if (!document) {
        return attachment;
      }
      return {
        ...attachment,
        display_name: document.display_name || attachment.display_name,
        status: document.status,
        metadata: {
          ...attachment.metadata,
          title: document.title,
          filename: document.filename
        }
      };
    })
);
const latestMessageSignature = computed(() => {
  const latestMessage = store.messages[store.messages.length - 1];
  return `${store.messages.length}:${latestMessage?.id || ""}:${latestMessage?.content.length || 0}:${store.sending}`;
});
const supportedCapabilityCommandIds = new Set<SlashCommandId>(["summary", "compare", "tag", "library", "help"]);
const stableCapabilityItems = computed(() => store.workbenchCapabilities?.stable_capabilities ?? []);
const confirmationCapabilityItems = computed(() => store.workbenchCapabilities?.confirmation_required ?? []);
const experimentalCapabilityItems = computed(() => store.workbenchCapabilities?.experimental_capabilities ?? []);
const capabilitySlashCommands = computed(() =>
  (store.workbenchCapabilities?.slash_commands ?? [])
    .filter((command) => supportedCapabilityCommandIds.has(command.id))
);
const traceSummary = computed(() => store.messageTraceSummary);
const latestAssistantMessage = computed(() =>
  [...store.messages].reverse().find((message) => message.role === "assistant") ?? null
);
const selectedTraceMessage = computed(() =>
  store.messages.find((message) => message.id === store.selectedTraceMessageId && message.role === "assistant") ?? null
);
const selectedTraceMessageLabel = computed(() => {
  const message = selectedTraceMessage.value;
  if (!message) {
    return "Assistant message";
  }
  return `PaperDesk · ${formatTime(message.created_at)}`;
});
const traceUsedDocuments = computed(() => {
  const ids = traceSummary.value?.used_document_ids ?? [];
  return ids.map((documentId) => {
    const document = workbenchDocuments.value.find((item) => item.id === documentId);
    return {
      id: documentId,
      label: document?.display_name || document?.filename || shortId(documentId),
      title: document?.title || document?.filename || documentId
    };
  });
});
const traceUsedSkills = computed(() => traceSummary.value?.used_skills ?? []);

onMounted(async () => {
  await Promise.all([documentStore.refreshDocuments(), store.bootstrap()]);
});

watch(latestMessageSignature, () => {
  void scrollMessagesToBottom();
}, { flush: "post" });

async function scrollMessagesToBottom() {
  await nextTick();
  if (messagePanelRef.value) {
    messagePanelRef.value.scrollTop = messagePanelRef.value.scrollHeight;
  }
}

function triggerImageInput() {
  showAttachMenu.value = false;
  imageInputRef.value?.click();
}

function triggerPdfInput() {
  showAttachMenu.value = false;
  pdfInputRef.value?.click();
}

function triggerSessionFileInput() {
  showAttachMenu.value = false;
  sessionFileInputRef.value?.click();
}

function toggleDocumentPicker() {
  showAttachMenu.value = false;
  showDocumentPicker.value = !showDocumentPicker.value;
}

async function handleImageSelected(event: Event) {
  const input = event.target as HTMLInputElement;
  const file = input.files?.[0];
  if (!file) {
    return;
  }
  const dataUrl = await readFileAsDataUrl(file);
  const attachment: ChatAttachment = {
    id: crypto.randomUUID(),
    kind: "image",
    display_name: file.name,
    mime_type: file.type,
    data_url: dataUrl,
    status: "ready",
    metadata: {
      size: file.size
    }
  };
  store.queueImageAttachment(attachment);
  input.value = "";
}

async function handlePdfSelected(event: Event) {
  const input = event.target as HTMLInputElement;
  const file = input.files?.[0];
  if (!file) {
    return;
  }
  try {
    const document = await documentStore.addDocument(file);
    if (document) {
      store.toggleLibraryDocument(document);
      store.markUploadedTaskDocument(document.id);
      await store.refreshWorkbenchFileContext();
    }
  } catch {
    store.queueLocalPdfAttachment(file);
  }
  input.value = "";
}

async function handleSessionFileSelected(event: Event) {
  const input = event.target as HTMLInputElement;
  const file = input.files?.[0];
  if (!file) {
    return;
  }
  const extension = file.name.split(".").pop()?.toLowerCase() || "";
  if (!allowedSessionFileExtensions.has(extension)) {
    store.sessionFileUploadError = "仅支持上传 .txt、.md、.docx 文件。";
    input.value = "";
    return;
  }
  if (file.size > sessionFileMaxBytes) {
    store.sessionFileUploadError = "文件超过 5MB，请选择更小的 txt、md 或 docx 文件。";
    input.value = "";
    return;
  }
  try {
    await store.uploadSessionFile(file);
  } catch {
    // Store owns the visible upload error.
  } finally {
    input.value = "";
  }
}

async function readFileAsDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result || ""));
    reader.onerror = () => reject(new Error("读取图片失败"));
    reader.readAsDataURL(file);
  });
}

function handleKeydown(event: KeyboardEvent) {
  if (store.slashCommandMenuOpen) {
    if (event.key === "ArrowDown") {
      event.preventDefault();
      moveSlashCommandSelection(1);
      return;
    }
    if (event.key === "ArrowUp") {
      event.preventDefault();
      moveSlashCommandSelection(-1);
      return;
    }
    if (event.key === "Escape") {
      event.preventDefault();
      store.slashCommandMenuOpen = false;
      return;
    }
    if (event.key === "Enter" && !event.shiftKey && showSlashMenu.value) {
      event.preventDefault();
      const command = filteredSlashCommands.value[slashCommandIndex.value];
      if (command) {
        selectSlashCommand(command);
      }
      return;
    }
  }
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    if (!store.sending && canSendMessage.value) {
      void sendChatMessage();
    }
  }
}

function handleComposerInput() {
  const hasSlashQuery = /(?:^|\s)\/[^\s]*$/.test(store.composerText);
  store.slashCommandMenuOpen = hasSlashQuery && !store.activeSlashCommand;
  if (store.slashCommandMenuOpen) {
    slashCommandIndex.value = 0;
  }
}

function moveSlashCommandSelection(direction: number) {
  const count = filteredSlashCommands.value.length;
  if (!count) {
    slashCommandIndex.value = 0;
    return;
  }
  slashCommandIndex.value = (slashCommandIndex.value + direction + count) % count;
}

function selectSlashCommand(command: SlashCommandOption) {
  store.applySlashCommand(command);
  slashCommandIndex.value = 0;
}

function applyCapabilitySlashCommand(capability: WorkbenchCapability) {
  if (!canApplyCapabilitySlashCommand(capability)) {
    return;
  }
  applySlashCommandById(capability.slash_command);
}

function applySlashCommandById(commandId?: SlashCommandId | null) {
  if (!commandId || !supportedCapabilityCommandIds.has(commandId)) {
    return;
  }
  const command = slashCommands.find((item) => item.id === commandId);
  if (command) {
    store.applySlashCommand(command);
  }
}

function canApplyCapabilitySlashCommand(capability: WorkbenchCapability) {
  return Boolean(
    capability.current_available &&
    capability.slash_command &&
    supportedCapabilityCommandIds.has(capability.slash_command)
  );
}

function startComposerResize(event: PointerEvent) {
  resizingComposer.value = true;
  resizeStartY.value = event.clientY;
  resizeStartHeight.value = composerHeight.value;
  window.addEventListener("pointermove", resizeComposer);
  window.addEventListener("pointerup", stopComposerResize);
  window.addEventListener("pointercancel", stopComposerResize);
}

function resizeComposer(event: PointerEvent) {
  if (!resizingComposer.value) {
    return;
  }
  const maxHeight = Math.min(Math.floor(window.innerHeight * 0.5), 420);
  const nextHeight = resizeStartHeight.value + resizeStartY.value - event.clientY;
  composerHeight.value = Math.max(92, Math.min(maxHeight, nextHeight));
}

function stopComposerResize() {
  resizingComposer.value = false;
  window.removeEventListener("pointermove", resizeComposer);
  window.removeEventListener("pointerup", stopComposerResize);
  window.removeEventListener("pointercancel", stopComposerResize);
}

onBeforeUnmount(() => {
  store.stopGeneration();
  stopComposerResize();
});

function handleComposerAction() {
  if (store.sending) {
    store.stopGeneration();
    return;
  }
  void sendChatMessage();
}

async function sendChatMessage() {
  const response = await store.sendCurrentMessage();
  if (response?.library_mutated) {
    await Promise.all([documentStore.refreshDocuments(), documentStore.refreshCategories()]);
    await store.refreshWorkbenchFileContext();
  }
}

function selectTraceMessage(message: ChatMessage) {
  if (message.role !== "assistant") {
    return;
  }
  void store.selectAssistantMessageForTrace(message);
}

function showMessageTrace(message: ChatMessage) {
  rightPanelTab.value = "trace";
  void store.selectAssistantMessageForTrace(message);
}

function activateTraceTab() {
  rightPanelTab.value = "trace";
  if (!store.selectedTraceMessageId && latestAssistantMessage.value) {
    void store.selectAssistantMessageForTrace(latestAssistantMessage.value);
  }
}

function toggleWorkbenchDocument(document: WorkbenchFileItem) {
  if (document.status !== "ready") {
    return;
  }
  store.toggleLibraryDocument(document);
}

function isWorkbenchDocumentSelected(documentId: string) {
  return store.selectedDocumentIds.includes(documentId);
}

function isWorkbenchDocumentUsed(documentId: string) {
  return workbenchUsedDocumentIds.value.has(documentId);
}

function isWorkbenchDocumentRecent(documentId: string) {
  return workbenchRecentDocumentIds.value.has(documentId);
}

function getSessionFileId(file: WorkbenchFileAsset) {
  return file.file_id || file.id;
}

function isSessionFileSelectable(file: WorkbenchFileAsset) {
  return file.status === "ready" && file.text_extract_status === "ready";
}

function toggleWorkbenchSessionFile(file: WorkbenchFileAsset) {
  if (!isSessionFileSelectable(file)) {
    return;
  }
  store.toggleSessionFile(getSessionFileId(file));
}

function isWorkbenchSessionFileSelected(file: WorkbenchFileAsset) {
  return store.isSessionFileSelected(getSessionFileId(file));
}

function isWorkbenchSessionFileUsed(file: WorkbenchFileAsset) {
  return workbenchUsedFileIds.value.has(getSessionFileId(file)) || workbenchUsedFileIds.value.has(file.id);
}

function isMessageProcessing(message: ChatMessage) {
  return message.role === "assistant" && (message.status === "processing" || (message.status === "streaming" && !message.content.trim()));
}

async function copyMessage(message: ChatMessage) {
  await navigator.clipboard.writeText(message.content);
  copiedMessageId.value = message.id;
  window.setTimeout(() => {
    if (copiedMessageId.value === message.id) {
      copiedMessageId.value = "";
    }
  }, 1600);
}

async function saveMessageReport(message: ChatMessage) {
  await store.saveAssistantMessageAsReport(message);
}

function formatTraceValue(value?: string | null) {
  if (!value) {
    return "None";
  }
  return value
    .split("_")
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function formatConfidence(value?: number | null) {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "0%";
  }
  return `${Math.round(Math.max(0, Math.min(1, value)) * 100)}%`;
}

function formatRiskLevel(value?: string | null) {
  return formatTraceValue(value || "unknown");
}

function formatConfirmationStatus(value?: string | null) {
  switch (value) {
    case "required":
      return "等待确认";
    case "executed":
      return "已执行";
    case "failed":
      return "执行失败";
    default:
      return "无确认";
  }
}

function shortId(value?: string | null) {
  if (!value) {
    return "";
  }
  return value.length <= 12 ? value : `${value.slice(0, 8)}...${value.slice(-4)}`;
}

function formatRole(role: ChatMessageRole) {
  switch (role) {
    case "assistant":
      return "PaperDesk";
    case "system":
      return "系统";
    default:
      return "你";
  }
}

function sessionSummary(session: ChatSession) {
  return session.title || session.last_message_preview || "从这里继续新的知识对话。";
}

function formatAttachmentKind(kind: string) {
  switch (kind) {
    case "image":
      return "图片";
    case "uploaded_pdf":
      return "本地 PDF";
    case "library_document":
      return "库内论文";
    default:
      return kind;
  }
}

function formatDocumentStatus(status: string) {
  switch (status) {
    case "ready":
      return "可用";
    case "processing":
      return "处理中";
    case "failed":
      return "失败";
    default:
      return status;
  }
}

function formatSessionFileStatus(status: string) {
  switch (status) {
    case "uploaded":
      return "已上传";
    case "ready":
      return "可预览";
    case "processing":
      return "处理中";
    case "failed":
      return "失败";
    case "unsupported":
      return "暂不支持";
    case "skipped":
      return "已跳过";
    default:
      return status;
  }
}

function formatTextExtractionStatus(status: string) {
  switch (status) {
    case "pending":
      return "待提取";
    case "ready":
      return "已提取";
    case "failed":
      return "失败";
    case "skipped":
      return "已跳过";
    default:
      return status;
  }
}

function formatFileKind(kind: string) {
  switch (kind) {
    case "txt":
      return "TXT";
    case "md":
      return "MD";
    case "docx":
      return "DOCX";
    default:
      return "FILE";
  }
}

function formatFileMeta(file: WorkbenchFileAsset) {
  const extension = file.extension ? file.extension.toUpperCase() : formatFileKind(file.kind);
  return `${extension} · ${formatBytes(file.size_bytes)} · ${formatTime(file.created_at)}`;
}

function formatBytes(value: number) {
  if (!Number.isFinite(value) || value <= 0) {
    return "0 B";
  }
  if (value < 1024) {
    return `${value} B`;
  }
  if (value < 1024 * 1024) {
    return `${(value / 1024).toFixed(1)} KB`;
  }
  return `${(value / 1024 / 1024).toFixed(1)} MB`;
}

function sessionFileReason(file: WorkbenchFileAsset) {
  if (file.failure_reason) {
    return file.failure_reason;
  }
  if (file.status === "unsupported" || file.text_extract_status === "skipped") {
    return "该文件当前仅保存展示，文本提取已跳过。";
  }
  if (file.status === "failed" || file.text_extract_status === "failed") {
    return "文件处理失败，请查看后端返回原因。";
  }
  return "";
}

function formatCapabilityIoType(value: string) {
  if (value === "write") {
    return "写入";
  }
  return "只读";
}

function formatOperationLevel(value: string) {
  switch (value) {
    case "query-level":
      return "查询级";
    case "entity-level":
      return "实体级";
    case "relation-level":
      return "关系级";
    case "content-level":
      return "内容级";
    default:
      return value;
  }
}

function formatTime(value: string) {
  return new Date(value).toLocaleString("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit"
  });
}

</script>
