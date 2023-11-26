<script setup>
import Transcriber from '../components/Transcriber.vue'
import Uploader from '../components/Uploader.vue'
import ChatWindowAudio from '../components/ChatWindowAudio.vue'

import { useAudioChatStore } from '../stores/audioChat.cjs'

const audioChatStore = useAudioChatStore()
</script>

<template>
  <article class="grid grid-cols-1 lg:grid-cols-2 gap-x-8">
    <div class="mt-4">
      <h3>Choose audio</h3>
      <!-- Uploader -->
      <section class="my-4">
        <Uploader />
      </section>
      <!-- Transcriber -->
      <Transcriber />
    </div>
    <div class="min-h-80 flex flex-col justify-between">
      <!-- Chat Window -->
      <ChatWindowAudio />
      <div class="flex justify-end">
        <button @click="audioChatStore.clearChat()" class="button button-secondary">Clear</button>
      </div>
    </div>
  </article>
</template>
